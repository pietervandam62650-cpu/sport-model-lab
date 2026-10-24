"""Évaluation chronologique : walk-forward (prévoir puis apprendre), Brier,
log loss, calibration, comparaison à une référence avec intervalle bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import ODDS_COLS
from .models import outcome_of

EPS = 1e-12


def walk_forward(df: pd.DataFrame, model, has_draw: bool) -> pd.DataFrame:
    """Pour chaque match, dans l'ordre : prévision avec l'état courant, PUIS mise à jour.
    Le modèle reste dans son état final (réutilisable pour prévoir des matchs futurs)."""
    rows = []
    for d, h, a, hs, as_ in zip(df["event_date"], df["home_team"], df["away_team"],
                                df["home_score"], df["away_score"]):
        ph, pdr, pa = model.predict(h, a)
        rows.append((d, h, a, ph, pdr, pa, outcome_of(hs, as_, has_draw)))
        model.update(h, a, hs, as_)
    out = pd.DataFrame(rows, columns=["event_date", "home_team", "away_team",
                                      "p_home", "p_draw", "p_away", "actual"])
    for c in ODDS_COLS:
        if c in df.columns:
            out[c] = df[c].to_numpy()
    return out


def _arrays(pred: pd.DataFrame, has_draw: bool):
    cols = ["p_home", "p_draw", "p_away"] if has_draw else ["p_home", "p_away"]
    labels = ["home", "draw", "away"] if has_draw else ["home", "away"]
    P = pred[cols].to_numpy(float)
    P = P / P.sum(axis=1, keepdims=True)
    y = np.array([labels.index(a) for a in pred["actual"]], dtype=int)
    return P, y


def per_match_scores(pred: pd.DataFrame, has_draw: bool):
    P, y = _arrays(pred, has_draw)
    brier = ((P - np.eye(P.shape[1])[y]) ** 2).sum(axis=1)
    logloss = -np.log(np.clip(P[np.arange(len(y)), y], EPS, 1.0))
    return brier, logloss, (P.argmax(axis=1) == y)


def sample_warning(n: int) -> str:
    if n < 30:
        return "Échantillon très insuffisant (< 30) : aucune conclusion possible."
    if n < 100:
        return "Échantillon insuffisant (< 100) : calibration et comparaisons non interprétables."
    return "Échantillon suffisant pour une première lecture ; reste une évaluation sur une seule période."


def metrics(pred: pd.DataFrame, has_draw: bool) -> dict:
    if len(pred) == 0:
        return {"n": 0, "brier": np.nan, "logloss": np.nan, "accuracy": np.nan}
    b, l, ok = per_match_scores(pred, has_draw)
    return {"n": int(len(pred)), "brier": float(b.mean()), "logloss": float(l.mean()),
            "accuracy": float(ok.mean())}


def calibration_table(pred: pd.DataFrame, has_draw: bool, bins: int = 10, min_n: int = 20) -> pd.DataFrame:
    parts = [("Victoire domicile", "p_home", "home"), ("Nul", "p_draw", "draw"),
             ("Victoire extérieur", "p_away", "away")]
    if not has_draw:
        parts = [parts[0], parts[2]]
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for label, col, target in parts:
        p = pred[col].to_numpy(float)
        obs = (pred["actual"] == target).to_numpy(float)
        idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
        for b in range(bins):
            m = idx == b
            if m.any():
                rows.append({"issue": label, "tranche": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                             "prob_moyenne": float(p[m].mean()), "freq_observee": float(obs[m].mean()),
                             "n": int(m.sum()), "fiable": bool(m.sum() >= min_n)})
    return pd.DataFrame(rows, columns=["issue", "tranche", "prob_moyenne", "freq_observee", "n", "fiable"])


def split_segments(n: int, warmup: float = 0.4, val: float = 0.3):
    """Trois tranches chronologiques : rodage (non évalué), validation (réglages), test final."""
    a = int(n * warmup)
    b = int(n * (warmup + val))
    return slice(0, a), slice(a, b), slice(b, n)


def compare(preds: dict, has_draw: bool, reference: str = "baseline_frequences",
            n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Tableau de comparaison. d_logloss < 0 = meilleur que la référence.
    L'IC95 est un bootstrap par paire de matchs (suppose les matchs échangeables : approximation)."""
    if reference not in preds:
        reference = next(iter(preds))
    scores = {k: per_match_scores(v, has_draw) for k, v in preds.items()}
    ref_ll = scores[reference][1]
    rng = np.random.default_rng(seed)
    rows = []
    for name, (b, l, ok) in scores.items():
        n = len(b)
        row = {"modele": name, "n": n, "brier": b.mean() if n else np.nan,
               "logloss": l.mean() if n else np.nan, "accuracy": ok.mean() if n else np.nan,
               "d_logloss": np.nan, "ic95_bas": np.nan, "ic95_haut": np.nan}
        if name == reference:
            row["verdict"] = "référence"
        elif n < 2:
            row["verdict"] = "données insuffisantes"
        else:
            diff = l - ref_ll
            idx = rng.integers(0, n, size=(n_boot, n))
            boots = diff[idx].mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            row.update(d_logloss=float(diff.mean()), ic95_bas=float(lo), ic95_haut=float(hi))
            if n < 100:
                row["verdict"] = "échantillon < 100 : non concluant"
            elif hi < 0:
                row["verdict"] = "meilleur que la référence sur cette période (non garanti ensuite)"
            elif lo > 0:
                row["verdict"] = "moins bon que la référence"
            else:
                row["verdict"] = "pas de différence détectable"
        rows.append(row)
    return pd.DataFrame(rows)


def implied_probs(pred: pd.DataFrame, has_draw: bool) -> pd.DataFrame:
    """Probabilités implicites des cotes, marge retirée par normalisation proportionnelle
    (méthode simple ; d'autres méthodes existent). Lignes sans cotes complètes : NaN."""
    cols = ["odds_home", "odds_draw", "odds_away"] if has_draw else ["odds_home", "odds_away"]
    if not all(c in pred.columns for c in cols):
        return pd.DataFrame(index=pred.index, columns=["p_home", "p_draw", "p_away"], dtype=float)
    inv = 1.0 / pred[cols].astype(float)
    norm = inv.div(inv.sum(axis=1), axis=0)
    out = pd.DataFrame(index=pred.index)
    out["p_home"] = norm["odds_home"]
    out["p_draw"] = norm["odds_draw"] if has_draw else 0.0
    out["p_away"] = norm["odds_away"]
    return out


def compare_with_bookmaker(preds: dict, has_draw: bool) -> pd.DataFrame | None:
    """Compare les modèles aux cotes normalisées, sur les seuls matchs avec cotes complètes."""
    any_pred = next(iter(preds.values()))
    book = implied_probs(any_pred, has_draw)
    mask = book[["p_home", "p_away"] + (["p_draw"] if has_draw else [])].notna().all(axis=1)
    if int(mask.sum()) == 0:
        return None
    sub = {k: v.loc[mask].reset_index(drop=True) for k, v in preds.items()}
    bk = sub[next(iter(sub))][["event_date", "home_team", "away_team", "actual"]].copy()
    bk[["p_home", "p_draw", "p_away"]] = book.loc[mask, ["p_home", "p_draw", "p_away"]].to_numpy()
    sub["bookmaker_normalise"] = bk
    return compare(sub, has_draw, reference="bookmaker_normalise")
