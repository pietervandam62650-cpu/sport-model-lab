import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from sports_lab.data import validate_results
from sports_lab.registry import Registry

NOW = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
LATER = NOW + timedelta(days=3)


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "sub" / "r.sqlite3"
        self.reg = Registry(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def add(self, **kw):
        base = dict(sport="football", home_team="A", away_team="B", event_date="2026-03-02",
                    probs=(0.5, 0.3, 0.2), model_version="elo-v0.2.0", now=NOW)
        base.update(kw)
        return self.reg.add_forecast(**base)

    def test_add_and_persist_across_instances(self):
        fid = self.add(sources="football-data", uncertainty="blessures inconnues", odds=(2.0, 3.4, 3.8),
                       odds_time="2026-03-01T09:00:00+00:00")
        again = Registry(self.path)  # nouvelle instance = simule un redémarrage
        df = again.list_forecasts()
        self.assertEqual(df.loc[0, "id"], fid)
        self.assertAlmostEqual(df.loc[0, "p_home"], 0.5)
        self.assertEqual(again.verify_integrity(), [])

    def test_validation_errors(self):
        for kw in [dict(probs=(0.5, 0.5, 0.5)), dict(probs=(1.2, -0.1, -0.1)), dict(home_team="B"),
                   dict(market="XX"), dict(market="ML", probs=(0.5, 0.2, 0.3)), dict(odds=(1.0, None, None)),
                   dict(event_date="2026-02-01")]:
            with self.assertRaises(ValueError, msg=str(kw)):
                self.add(**kw)

    def test_backfill_flagged_and_excluded_from_evaluation(self):
        fid = self.add(event_date="2026-02-01", backfill=True)
        self.reg.add_result(fid, 1, 0, now=NOW)
        self.assertTrue(self.reg.evaluate(now=NOW).empty)
        self.assertEqual(len(self.reg.evaluate(include_backfill=True, now=NOW)), 1)

    def test_immutability_triggers(self):
        fid = self.add()
        self.reg.add_result(fid, 2, 1, now=LATER)
        with sqlite3.connect(self.path) as con:
            for sql in ["UPDATE forecasts SET p_home=0.9", "DELETE FROM forecasts",
                        "UPDATE results SET home_score=9", "DELETE FROM results"]:
                with self.assertRaises(sqlite3.IntegrityError, msg=sql):
                    con.execute(sql)

    def test_tampering_detected_by_hash_and_missing_trigger(self):
        fid = self.add()
        con = sqlite3.connect(self.path)
        con.execute("DROP TRIGGER forecasts_no_update")
        con.execute("UPDATE forecasts SET p_home=0.9, p_draw=0.05, p_away=0.05 WHERE id=?", (fid,))
        con.commit()
        con.close()
        issues = self.reg.verify_integrity()
        self.assertTrue(any("hachage invalide" in i for i in issues))
        self.assertTrue(any("Trigger" in i for i in issues))

    def test_result_rules(self):
        fid = self.add()
        with self.assertRaises(ValueError):
            self.reg.add_result(fid, 1, 0, now=NOW)  # match pas encore joué
        out = self.reg.add_result(fid, 2, 1, now=LATER)
        self.assertEqual(out["outcome"], "home")
        self.assertAlmostEqual(out["brier"], 0.25 + 0.09 + 0.04, places=9)  # issue = domicile
        self.assertAlmostEqual(out["log_loss"], -__import__("math").log(0.5), places=9)
        with self.assertRaises(ValueError):
            self.reg.add_result(fid, 3, 3, now=LATER)  # second résultat refusé
        ml = self.add(market="ML", probs=(0.6, 0.0, 0.4))
        with self.assertRaises(ValueError):
            self.reg.add_result(ml, 80, 80, now=LATER)  # nul impossible

    def test_correction_is_a_new_record(self):
        first = self.add()
        second = self.add(probs=(0.4, 0.3, 0.3), supersedes_id=first)
        df = self.reg.list_forecasts().set_index("id")
        self.assertAlmostEqual(df.loc[first, "p_home"], 0.5)  # l'original est inchangé
        self.assertEqual(df.loc[second, "supersedes_id"], first)
        with self.assertRaises(ValueError):
            self.add(supersedes_id="inconnu")

    def test_evaluate_matches_manual_computation_and_warns_small_n(self):
        a = self.add(probs=(0.6, 0.3, 0.1))
        b = self.add(probs=(0.2, 0.3, 0.5), home_team="C", away_team="D")
        self.reg.add_result(a, 1, 0, now=LATER)
        self.reg.add_result(b, 0, 1, now=LATER)
        ev = self.reg.evaluate(now=LATER)
        self.assertEqual(int(ev.loc[0, "n"]), 2)
        self.assertAlmostEqual(ev.loc[0, "brier"], ((0.16 + 0.09 + 0.01) + (0.04 + 0.09 + 0.25)) / 2, places=9)
        self.assertAlmostEqual(ev.loc[0, "brier_uniforme"], 2 / 3, places=9)
        self.assertIn("très insuffisant", ev.loc[0, "avertissement"])

    def test_export_import_roundtrip_and_rejects_tampered(self):
        fid = self.add()
        self.reg.add_result(fid, 1, 1, now=LATER)
        dump = self.reg.export_json()
        other = Registry(Path(self.tmp.name) / "other.sqlite3")
        stats = other.import_json(dump)
        self.assertEqual((stats["forecasts_added"], stats["results_added"]), (1, 1))
        self.assertEqual(other.verify_integrity(), [])
        self.assertEqual(other.import_json(dump)["forecasts_skipped"], 1)  # idempotent
        bad = json.loads(dump)
        bad["forecasts"][0]["p_home"] = 0.99
        stats = Registry(Path(self.tmp.name) / "third.sqlite3").import_json(json.dumps(bad))
        self.assertEqual(stats["forecasts_added"], 0)
        self.assertEqual(stats["results_added"], 0)
        self.assertEqual(len(stats["rejected"]), 2)  # la prévision altérée et son résultat

    def test_dataset_store_roundtrip(self):
        raw = pd.DataFrame({"sport": ["football"], "event_date": ["2025-01-01"], "home_team": ["A"],
                            "away_team": ["B"], "home_score": [1], "away_score": [0]})
        clean, _ = validate_results(raw)
        self.reg.save_dataset("test", clean, "unit")
        back, rep = Registry(self.path).load_dataset("test")
        self.assertEqual(rep["rows_out"], 1)
        pd.testing.assert_frame_equal(back, clean, check_dtype=False)
        self.assertEqual(self.reg.list_datasets().loc[0, "n_rows"], 1)


if __name__ == "__main__":
    unittest.main()
