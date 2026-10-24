import unittest

import numpy as np

from sports_lab.evaluation import walk_forward
from sports_lab.models import EloModel, FrequencyBaseline, PoissonModel, UniformBaseline, build_models, outcome_of
from tests.helpers import synthetic_football


class ModelTests(unittest.TestCase):
    def test_outcome_of(self):
        self.assertEqual(outcome_of(2, 1, True), "home")
        self.assertEqual(outcome_of(1, 1, True), "draw")
        with self.assertRaises(ValueError):
            outcome_of(1, 1, False)

    def test_probabilities_sum_to_one(self):
        for hd in (True, False):
            for m in (UniformBaseline(hd), FrequencyBaseline(hd), EloModel(hd), ):
                self.assertAlmostEqual(sum(m.predict("A", "B")), 1.0, places=9)
        self.assertAlmostEqual(sum(PoissonModel().predict("A", "B")), 1.0, places=9)

    def test_elo_draw_rate_at_equal_strength(self):
        m = EloModel(True, home_adv=0.0, draw_prior=0.25)
        ph, pdr, pa = m.predict("A", "B")
        self.assertAlmostEqual(pdr, 0.25, places=9)
        self.assertAlmostEqual(ph, pa, places=9)

    def test_elo_two_way_has_no_draw_and_matches_logistic(self):
        m = EloModel(False, home_adv=0.0)
        m.ratings = {"A": 1700.0, "B": 1500.0}
        ph, pdr, pa = m.predict("A", "B")
        self.assertEqual(pdr, 0.0)
        self.assertAlmostEqual(ph, 1 / (1 + 10 ** (-200 / 400)), places=9)

    def test_elo_update_direction_and_zero_sum(self):
        m = EloModel(True)
        m.update("A", "B", 3, 0)
        self.assertGreater(m.ratings["A"], 1500)
        self.assertAlmostEqual(m.ratings["A"] + m.ratings["B"], 3000.0, places=9)

    def test_poisson_markets_consistent(self):
        m = PoissonModel()
        for _ in range(30):
            m.update("A", "B", 3, 0)
        mk = m.markets("A", "B")
        self.assertGreater(mk["home"], mk["away"])
        self.assertTrue(0 <= mk["over_2_5"] <= 1 and 0 <= mk["btts"] <= 1)
        self.assertAlmostEqual(mk["home"] + mk["draw"] + mk["away"], 1.0, places=9)

    def test_build_models_per_sport(self):
        self.assertIn("poisson_online", build_models("football", True))
        self.assertNotIn("poisson_online", build_models("basketball", False))
        self.assertNotIn("poisson_online", build_models("football", False))
        self.assertEqual(build_models("tennis", False)["elo"].home_adv, 0.0)


class NoLeakageTests(unittest.TestCase):
    """Les prévisions du match i ne doivent pas changer si les résultats FUTURS changent."""

    def test_predictions_independent_of_future_results(self):
        df = synthetic_football(n_teams=6, rounds=4)
        cut = len(df) // 2
        altered = df.copy()
        rng = np.random.default_rng(1)
        altered.loc[cut:, "home_score"] = rng.integers(0, 6, len(df) - cut)
        altered.loc[cut:, "away_score"] = rng.integers(0, 6, len(df) - cut)
        for name in build_models("football", True):
            a = walk_forward(df, build_models("football", True)[name], True)
            b = walk_forward(altered, build_models("football", True)[name], True)
            cols = ["p_home", "p_draw", "p_away"]
            np.testing.assert_allclose(a.loc[:cut, cols].to_numpy(), b.loc[:cut, cols].to_numpy(),
                                       err_msg=f"fuite détectée pour {name}")
            if name != "baseline_uniforme":  # la référence uniforme ne dépend d'aucune donnée
                self.assertFalse(np.allclose(a.loc[cut + 5:, cols].to_numpy(), b.loc[cut + 5:, cols].to_numpy()),
                                 f"le test n'est pas discriminant pour {name}")


if __name__ == "__main__":
    unittest.main()
