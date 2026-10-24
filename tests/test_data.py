import unittest
from unittest import mock
from urllib.error import URLError

import pandas as pd

from sports_lab.data import DataError, prepare_sport, validate_results
from sports_lab.sources import SourceError, fetch_football_data, football_data_url, parse_football_data_csv

FD = ("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,B365H,B365D,B365A\n"
      "E0,16/08/2024,Man United,Fulham,1,0,1.5,4.5,6.5\n"
      "E0,17/08/24,Arsenal,Wolves,2,0,1.2,7,13\n"
      "E0,18/08/2024,Chelsea,Brentford,,,1.9,3.6,4\n")


class ValidateTests(unittest.TestCase):
    def test_missing_columns(self):
        with self.assertRaises(DataError):
            validate_results(pd.DataFrame({"sport": ["x"]}))

    def test_drops_are_counted_not_silent(self):
        raw = pd.DataFrame({
            "sport": ["Football", "football", "football", "football", "football", "football"],
            "event_date": ["2025-01-01", "2025-01-01", "pas une date", "2025-01-02", "2025-01-03", "2025-01-04"],
            "home_team": ["A", "A", "B", "C", "D", "E"], "away_team": ["B", "B", "C", "C", "E", "F"],
            "home_score": [1, 1, 1, 1, -1, 1.5], "away_score": [0, 0, 0, 0, 0, 0]})
        clean, rep = validate_results(raw)
        self.assertEqual(len(clean), 1)
        self.assertEqual(rep["rows_in"], 6)
        self.assertEqual(sum(rep["dropped"].values()), 5)
        self.assertEqual(clean.loc[0, "sport"], "football")

    def test_odds_invalid_become_nan_and_columns_always_exist(self):
        raw = pd.DataFrame({"sport": ["f"], "event_date": ["2025-01-01"], "home_team": ["A"], "away_team": ["B"],
                            "home_score": [1], "away_score": [0], "odds_home": [0.9]})
        clean, _ = validate_results(raw)
        self.assertTrue(clean["odds_home"].isna().all())
        self.assertIn("odds_draw", clean.columns)

    def test_chronological_stable_order(self):
        raw = pd.DataFrame({"sport": ["f"] * 3, "event_date": ["2025-01-03", "2025-01-01", "2025-01-01"],
                            "home_team": ["A", "B", "C"], "away_team": ["X", "Y", "Z"],
                            "home_score": [1, 1, 1], "away_score": [0, 0, 0]})
        clean, _ = validate_results(raw)
        self.assertEqual(list(clean["home_team"]), ["B", "C", "A"])

    def test_no_draw_sport_drops_ties(self):
        raw = pd.DataFrame({"sport": ["basketball"] * 2, "event_date": ["2025-01-01", "2025-01-02"],
                            "home_team": ["A", "A"], "away_team": ["B", "B"],
                            "home_score": [90, 90], "away_score": [90, 80]})
        clean, _ = validate_results(raw)
        sd, has_draw, notes = prepare_sport(clean, "basketball")
        self.assertFalse(has_draw)
        self.assertEqual(len(sd), 1)
        self.assertTrue(notes)


class SourceTests(unittest.TestCase):
    def test_parse_fixture_formats_and_missing_scores(self):
        out = parse_football_data_csv(FD, competition="PL")
        clean, rep = validate_results(out)
        self.assertEqual(len(clean), 2)  # le match sans score est écarté
        self.assertEqual(rep["dropped"], {"score manquant ou non numérique": 1})
        self.assertEqual(clean.loc[0, "event_date"], pd.Timestamp("2024-08-16"))
        self.assertEqual(clean.loc[1, "event_date"], pd.Timestamp("2024-08-17"))
        self.assertAlmostEqual(clean.loc[0, "odds_home"], 1.5)

    def test_missing_source_columns(self):
        with self.assertRaises(DataError):
            parse_football_data_csv("a,b\n1,2\n")

    def test_latin1_bytes(self):
        raw = FD.replace("Man United", "Sporting Gijón").encode("latin-1")
        self.assertIn("Sporting Gijón", parse_football_data_csv(raw)["home_team"].tolist())

    def test_url_validation(self):
        self.assertTrue(football_data_url("2425", "E0").endswith("/2425/E0.csv"))
        for bad in [("2024", "E0"), ("2427", "E0"), ("2425", "../x"), ("abcd", "E0")]:
            with self.assertRaises(SourceError):
                football_data_url(*bad)

    def test_fetch_success_and_network_failure_with_fake_opener(self):
        class Resp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self, n=-1): return FD.encode()
        df = fetch_football_data("2425", "E0", opener=lambda req, timeout: Resp())
        self.assertEqual(len(df), 3)
        self.assertEqual(df["competition"].iloc[0], "Premier League (Angleterre) 2425")
        def boom(req, timeout): raise URLError("hors ligne")
        with self.assertRaises(SourceError):
            fetch_football_data("2425", "E0", opener=boom)


if __name__ == "__main__":
    unittest.main()
