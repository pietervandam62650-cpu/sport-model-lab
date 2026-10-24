"""Enchaînements de haut niveau utilisés par l'interface (testables sans Streamlit)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import sport_config
from .evaluation import (calibration_table, compare, compare_with_bookmaker, metrics,
                         sample_warning, split_segments, walk_forward)
from .models import EloModel, PoissonModel, build_models

MIN_MATCHES = 20


def analyze_sport(sd: pd.DataFrame, sport: str, has_draw: bool, warmup: float = 0.4,
                  val: float = 0.3, tune: bool = True) -> dict:
    cfg = sport_config(sport)
    n = len(sd)
    res = {"sport": sport, "n": n, "has_draw": has_draw, "warnings": []}
    if n < MIN_MATCHES:
        res["error"] = f"Seulement {n} matchs : minimum {MIN_MATCHES} pour lancer une évaluation."
        return res
    _, v_sl, t_sl = split_segments(n, warmup, val)
    n_val, n_test = len(range(n)[v_sl]), len(range(n)[t_sl])

    elo_params = {"k": cfg["k"], "home_adv": cfg["home_adv"]}
    tuning = None
    if tune and n_val >= 20:
        ha_grid = (0.0,) if cfg["home_adv"] == 0 else (0.0, 30.0, 55.0, 80.0)
        rows = []
        for k in (10.0, 20.0, 30.0, 40.0):
            for ha in ha_grid:
                p = walk_forward(sd, EloModel(has_draw, k=k, home_adv=ha), has_draw)
                rows.append({"k": k, "home_adv": ha, "logloss_validation": metrics(p.iloc[v_sl], has_draw)["logloss"]})
        tuning = pd.DataFrame(rows).sort_values("logloss_validation").reset_index(drop=True)
        elo_params = {"k": float(tuning.loc[0, "k"]), "home_adv": float(tuning.loc[0, "home_adv"])}
        res["warnings"].append("Paramètres Elo choisis sur la tranche de validation uniquement ; "
                               "le test final n'a pas servi à les régler.")
    else:
        res["warnings"].append("Pas de réglage : validation trop courte, paramètres par défaut (non validés).")

    models = build_models(sport, has_draw, elo_params=elo_params)
    preds = {name: walk_forward(sd, m, has_draw) for name, m in models.items()}
    res.update(
        models=models, preds=preds, elo_params=elo_params, tuning=tuning,
        n_validation=n_val, n_test=n_test,
        validation=compare({k: v.iloc[v_sl] for k, v in preds.items()}, has_draw),
        test=compare({k: v.iloc[t_sl] for k, v in preds.items()}, has_draw),
        calibration={k: calibration_table(v.iloc[t_sl], has_draw)
                     for k, v in preds.items() if not k.startswith("baseline")},
        bookmaker=compare_with_bookmaker({k: v.iloc[t_sl] for k, v in preds.items()}, has_draw),
    )
    res["warnings"].append(sample_warning(n_test) + f" (test : {n_test} matchs)")
    if has_draw and "poisson_online" in models:
        res["warnings"].append("Poisson indépendant : tend à sous-estimer les nuls. Modèle expérimental.")
    if sport_config(sport)["draws"] is None:
        res["warnings"].append("Sport sans configuration dédiée : Elo générique, non adapté à ce sport par défaut.")
    return res


def forecast(model, home: str, away: str) -> dict:
    """Probabilités d'un match à venir avec l'état courant du modèle."""
    ph, pdr, pa = model.predict(home, away)
    out = {"p_home": ph, "p_draw": pdr, "p_away": pa}
    if isinstance(model, PoissonModel):
        out["extras"] = {k: v for k, v in model.markets(home, away).items() if k in ("over_2_5", "btts")}
    return out
