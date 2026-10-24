"""Générateur de données synthétiques reproductibles (graine fixe)."""
import numpy as np
import pandas as pd


def synthetic_football(n_teams=10, rounds=14, seed=7, strength_sd=0.35):
    rng = np.random.default_rng(seed)
    teams = [f"T{i:02d}" for i in range(n_teams)]
    att = rng.normal(0, strength_sd, n_teams)
    dfn = rng.normal(0, strength_sd, n_teams)
    rows, day = [], pd.Timestamp("2020-01-01")
    for _ in range(rounds):
        for i in range(n_teams):
            for j in range(n_teams):
                if i == j:
                    continue
                lh = 1.45 * np.exp(att[i] - dfn[j])
                la = 1.15 * np.exp(att[j] - dfn[i])
                rows.append(("football", day.strftime("%Y-%m-%d"), teams[i], teams[j],
                             int(rng.poisson(lh)), int(rng.poisson(la))))
                day += pd.Timedelta(hours=8)
    return pd.DataFrame(rows, columns=["sport", "event_date", "home_team", "away_team", "home_score", "away_score"])
