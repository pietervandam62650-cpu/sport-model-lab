import math
import unittest

import numpy as np
import pandas as pd

from sports_lab.config import sport_config
from sports_lab.evaluation import (calibration_table, compare, compare_with_bookmaker, implied_probs,
                                   metrics, split_segments, walk_forward)
from sports_lab.models import build_models
from sports_lab.workflows import analyze_sport, forecast
from tests.helpers import synthetic_football


def frame(probs, actual, odds=None):
    df = pd.DataFrame(probs, columns=["p_home", "p_draw", "p_away"])
    df["actual"] = actual
    df["event_date"], df["home_team"], df["away_team"] = pd.Timestamp("2025-01-01"), "A", "B"
    if odds is not None:
        df[["odds_home", "odds_draw", "odds_away"]] = odds
    return df


class MetricTests(unittest.TestCase):
    def test_uniform_three_way_known_values(self):
        pred = frame([[1/3, 1/3, 1/3]] * 3, ["home", "draw", "away"])
        m = metrics(pred, True)
        self.assertAlmostEqual(m["brier"], 2 / 3, places=9)
        self.assertAlmostEqual(m["logloss"], math.log(3), places=9)

    def test_hand_computed_brier_and_logloss(self):
        pred = frame([[0.6, 0.3, 0.1]], ["home"])
        m = metrics(pred, True)
        self.assertAlmostEqual(m["brier"], 0.4**2 + 0.3**2 + 0.1**2, places=9)
        self.assertAlmostEqual(m["logloss"], -math.log(0.6), places=9)

    def test_two_way_uniform(self):
        pred = frame([[0.5, 0, 0.5]] * 2, ["home", "away"])
        m = metrics(pred, False)
        self.assertAlmostEqual(m["brier"], 0.5, places=9)
        self.assertAlmostEqual(m["logloss"], math.log(2), places=9)

    def test_calibration_counts_and_values(self):
        pred = frame([[0.7, 0.2, 0.1]] * 10, ["home"] * 7 + ["away"] * 3)
        cal = calibration_table(pred, True)
        row = cal[(cal.issue == "Victoire domicile")].iloc[0]
        self.assertEqual(row["n"], 10)
        self.assertAlmostEqual(row["freq_observee"], 0.7)
        self.assertFalse(row["fiable"])  # n < 20
        self.assertEqual(int(cal[cal.issue == "Nul"]["n"].sum()), 10)

    def test_segments_are_chronological_and_disjoint(self):
        w, v, t = split_segments(100)
        self.assertEqual((w.stop, v.start, v.stop, t.start, t.stop), (40, 40, 70, 70, 100))

    def test_implied_probs_remove_margin(self):
        pred = frame([[1/3, 1/3, 1/3]], ["home"], odds=[[2.0, 3.5, 4.0]])
        ip = implied_probs(pred, True)
        self.assertAlmostEqual(float(ip.iloc[0].sum()), 1.0, places=9)
        raw = 1 / 2.0 + 1 / 3.5 + 1 / 4.0
        self.assertGreater(raw, 1.0)  # marge positive
        self.assertAlmostEqual(ip.loc[0, "p_home"], (1 / 2.0) / raw, places=9)

    def test_bookmaker_comparison_skips_rows_without_odds(self):
        a = frame([[0.5, 0.25, 0.25]] * 4, ["home"] * 4, odds=[[2, 3.5, 4]] * 2 + [[np.nan] * 3] * 2)
        res = compare_with_bookmaker({"m": a}, True)
        self.assertEqual(int(res.loc[res.modele == "m", "n"].iloc[0]), 2)
        self.assertIsNone(compare_with_bookmaker({"m": a.assign(odds_home=np.nan)}, True))

    def test_compare_small_sample_never_claims_superiority(self):
        good = frame([[0.9, 0.05, 0.05]] * 10, ["home"] * 10)
        ref = frame([[1/3, 1/3, 1/3]] * 10, ["home"] * 10)
        res = compare({"baseline_frequences": ref, "x": good}, True)
        row = res[res.modele == "x"].iloc[0]
        self.assertLess(row["d_logloss"], 0)
        self.assertIn("< 100", row["verdict"])


class SyntheticEndToEnd(unittest.TestCase):
    """Sur données SIMULÉES avec de vraies différences de force, les modèles à forces
    d'équipes doivent battre la référence. Cela valide le code, pas une performance réelle."""

    @classmethod
    def setUpClass(cls):
        cls.df = synthetic_football(n_teams=10, rounds=14, seed=7)  # 1260 matchs
        cls.res = analyze_sport(cls.df, "football", True)

    def test_structure(self):
        r = self.res
        self.assertEqual(r["n"], len(self.df))
        self.assertEqual(set(r["models"]), {"baseline_uniforme", "baseline_frequences", "elo", "poisson_online"})
        self.assertEqual(r["n_test"], len(self.df) - int(len(self.df) * 0.7))
        self.assertIsNotNone(r["tuning"])
        self.assertIsNone(r["bookmaker"])

    def test_models_beat_baseline_on_test(self):
        t = self.res["test"].set_index("modele")
        for name in ("elo", "poisson_online"):
            self.assertLess(t.loc[name, "logloss"], t.loc["baseline_frequences", "logloss"], name)
            self.assertIn("meilleur", t.loc[name, "verdict"])
        self.assertLess(t.loc["baseline_frequences", "logloss"], 1.2)

    def test_results_are_reproducible(self):
        again = analyze_sport(self.df, "football", True)
        pd.testing.assert_frame_equal(self.res["test"], again["test"])

    def test_forecast_uses_final_state(self):
        f = forecast(self.res["models"]["poisson_online"], "T00", "T01")
        self.assertAlmostEqual(f["p_home"] + f["p_draw"] + f["p_away"], 1.0, places=9)
        self.assertIn("over_2_5", f["extras"])

    def test_tuning_ignores_test_segment(self):
        """Modifier uniquement les résultats du segment test ne doit pas changer les paramètres choisis."""
        alt = self.df.copy()
        cut = int(len(alt) * 0.7)
        alt.loc[cut:, "home_score"] = 0
        alt.loc[cut:, "away_score"] = 5
        again = analyze_sport(alt, "football", True)
        self.assertEqual(again["elo_params"], self.res["elo_params"])


class SmallAndNoDraw(unittest.TestCase):
    def test_too_few_matches(self):
        r = analyze_sport(synthetic_football(3, 1), "football", True)
        self.assertIn("error", r)

    def test_no_draw_sport_pipeline(self):
        df = synthetic_football(8, 10)
        df = df[df.home_score != df.away_score].reset_index(drop=True)
        r = analyze_sport(df, "basketball", False)
        self.assertNotIn("poisson_online", r["models"])
        self.assertTrue((r["preds"]["elo"]["p_draw"] == 0).all())
        self.assertNotIn("Nul", set(r["calibration"]["elo"]["issue"]))

    def test_tennis_has_no_home_advantage_grid(self):
        self.assertEqual(sport_config("tennis")["home_adv"], 0.0)
        df = synthetic_football(8, 10)
        df = df[df.home_score != df.away_score].reset_index(drop=True)
        r = analyze_sport(df, "tennis", False)
        self.assertEqual(set(r["tuning"]["home_adv"]), {0.0})


if __name__ == "__main__":
    unittest.main()
