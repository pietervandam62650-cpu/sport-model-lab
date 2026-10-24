import unittest
from pathlib import Path

import pandas as pd

from sports_lab.data import prepare_sport, validate_results
from sports_lab.workflows import analyze_sport

SAMPLE = Path(__file__).resolve().parent.parent / "sample_results.csv"


class SampleFileTests(unittest.TestCase):
    def test_sample_pipeline_and_honest_warnings(self):
        clean, rep = validate_results(pd.read_csv(SAMPLE))
        self.assertEqual((rep["rows_in"], rep["rows_out"], rep["dropped"]), (40, 40, {}))
        sd, hd, _ = prepare_sport(clean, "football")
        res = analyze_sport(sd, "football", hd)
        self.assertNotIn("error", res)
        self.assertTrue(any("très insuffisant" in w for w in res["warnings"]))
        self.assertTrue((res["test"].loc[res["test"].modele != "baseline_frequences", "verdict"]
                         .str.contains("< 100|référence|insuffisantes")).all())
        sd, hd, _ = prepare_sport(clean, "basketball")
        self.assertIn("error", analyze_sport(sd, "basketball", hd))  # 10 matchs : refus explicite


if __name__ == "__main__":
    unittest.main()
