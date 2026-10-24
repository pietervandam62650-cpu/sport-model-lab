"""Validation et préparation des données de résultats."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import sport_config

REQUIRED = ["sport", "event_date", "home_team", "away_team", "home_score", "away_score"]
ODDS_COLS = ["odds_home", "odds_draw", "odds_away"]
OUTPUT_COLS = REQUIRED + ["competition"] + ODDS_COLS


class DataError(ValueError):
    """Données inutilisables (colonnes absentes, fichier illisible...)."""


def _clean_text(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.replace(r"\s+", " ", regex=True)


def validate_results(raw: pd.DataFrame):
    """Retourne (df_propre, rapport). Aucune ligne n'est corrigée en silence :
    chaque ligne écartée est comptée par motif dans rapport['dropped']."""
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise DataError("Colonnes manquantes : " + ", ".join(missing))

    report = {"rows_in": int(len(raw)), "dropped": {}, "warnings": []}
    x = raw.copy()
    x["_row"] = np.arange(len(x))

    def drop(frame, mask, reason):
        n = int(mask.sum())
        if n:
            report["dropped"][reason] = report["dropped"].get(reason, 0) + n
        return frame.loc[~mask].copy()

    x["sport"] = _clean_text(x["sport"]).str.casefold()
    x["home_team"] = _clean_text(x["home_team"])
    x["away_team"] = _clean_text(x["away_team"])
    x["competition"] = _clean_text(x["competition"]) if "competition" in x.columns else ""
    x = drop(x, (x["sport"] == "") | (x["home_team"] == "") | (x["away_team"] == ""),
             "sport ou équipe manquant")

    x["event_date"] = pd.to_datetime(x["event_date"], errors="coerce", utc=True).dt.tz_localize(None)
    x = drop(x, x["event_date"].isna(), "date invalide (format attendu : AAAA-MM-JJ)")

    x["home_score"] = pd.to_numeric(x["home_score"], errors="coerce")
    x["away_score"] = pd.to_numeric(x["away_score"], errors="coerce")
    x = drop(x, x["home_score"].isna() | x["away_score"].isna(), "score manquant ou non numérique")
    x = drop(x, (x["home_score"] < 0) | (x["away_score"] < 0), "score négatif")
    x = drop(x, (x["home_score"] % 1 != 0) | (x["away_score"] % 1 != 0), "score non entier")
    x = drop(x, x["home_team"] == x["away_team"], "même équipe des deux côtés")

    for c in ODDS_COLS:
        x[c] = pd.to_numeric(x[c], errors="coerce") if c in x.columns else np.nan
        x.loc[x[c] <= 1.0, c] = np.nan  # cote décimale <= 1 : invalide

    x = x.sort_values(["event_date", "_row"], kind="stable")
    x = drop(x, x.duplicated(["sport", "event_date", "home_team", "away_team"], keep="first"),
             "doublon (même sport, date, équipes)")
    x = x.sort_values(["event_date", "_row"], kind="stable").reset_index(drop=True)
    x = x[OUTPUT_COLS].copy()
    x[["home_score", "away_score"]] = x[["home_score", "away_score"]].astype(int)

    report["rows_out"] = int(len(x))
    if len(x) == 0:
        report["warnings"].append("Aucune ligne valide.")
    return x, report


def prepare_sport(df: pd.DataFrame, sport: str):
    """Extrait un sport. Retourne (df_sport, has_draw, notes)."""
    cfg = sport_config(sport)
    sd = df[df["sport"] == sport].reset_index(drop=True)
    ties = sd["home_score"] == sd["away_score"]
    notes = []
    has_draw = bool(ties.any()) if cfg["draws"] is None else bool(cfg["draws"])
    if cfg["draws"] is None:
        notes.append("Sport inconnu : le nul est considéré possible seulement si les données en contiennent.")
    if not has_draw and ties.any():
        notes.append(f"{int(ties.sum())} égalité(s) écartée(s) : ce sport est modélisé sans nul.")
        sd = sd.loc[~ties].reset_index(drop=True)
    return sd, has_draw, notes
