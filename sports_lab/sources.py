"""Sources de données publiques.

Source implémentée : football-data.co.uk (résultats CSV de ligues de football,
cotes de bookmakers en option). Limites à connaître :
- football uniquement ; pas de xG, pas de blessures, pas de compositions ;
- l'heure de relevé des cotes n'est pas fournie : elles servent de repère
  historique, pas de preuve de cote réellement disponible avant le match ;
- vérifier les conditions d'utilisation du site avant tout usage répété ;
- disponibilité et format non garantis (non testé en direct dans ce projet).
Les autres sports passent par l'import CSV.
"""
from __future__ import annotations

import io
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .data import DataError

LEAGUES = {
    "E0": "Premier League (Angleterre)",
    "E1": "Championship (Angleterre)",
    "F1": "Ligue 1 (France)",
    "F2": "Ligue 2 (France)",
    "D1": "Bundesliga (Allemagne)",
    "I1": "Serie A (Italie)",
    "SP1": "La Liga (Espagne)",
}
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
MAX_BYTES = 5_000_000


class SourceError(RuntimeError):
    """Échec réseau ou contenu inattendu pour une source en ligne."""


def football_data_url(season: str, league: str) -> str:
    season = str(season).strip()
    if not re.fullmatch(r"\d{4}", season) or (int(season[:2]) + 1) % 100 != int(season[2:]):
        raise SourceError("Saison invalide : utiliser par ex. 2425 pour 2024/2025.")
    if league not in LEAGUES:
        raise SourceError("Championnat non pris en charge : " + str(league))
    return BASE_URL.format(season=season, league=league)


def parse_football_data_csv(content, competition: str = "") -> pd.DataFrame:
    if isinstance(content, bytes):
        try:
            content = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            content = content.decode("latin-1")
    try:
        src = pd.read_csv(io.StringIO(content), on_bad_lines="skip")
    except Exception as exc:
        raise DataError(f"CSV illisible : {exc}") from exc
    need = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing = [c for c in need if c not in src.columns]
    if missing:
        raise DataError("Colonnes absentes du fichier source : " + ", ".join(missing))
    dates = pd.to_datetime(src["Date"], dayfirst=True, errors="coerce", format="mixed")
    out = pd.DataFrame({
        "sport": "football",
        "event_date": dates.dt.strftime("%Y-%m-%d"),
        "home_team": src["HomeTeam"],
        "away_team": src["AwayTeam"],
        "home_score": src["FTHG"],
        "away_score": src["FTAG"],
        "competition": competition,
        "odds_home": src["B365H"] if "B365H" in src.columns else np.nan,
        "odds_draw": src["B365D"] if "B365D" in src.columns else np.nan,
        "odds_away": src["B365A"] if "B365A" in src.columns else np.nan,
    })
    return out


def fetch_football_data(season: str, league: str, timeout: float = 20.0, opener=urlopen) -> pd.DataFrame:
    url = football_data_url(season, league)
    req = Request(url, headers={"User-Agent": "sports-model-lab/0.2 (analyse statistique, usage personnel)"})
    try:
        with opener(req, timeout=timeout) as resp:
            body = resp.read(MAX_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise SourceError(f"Téléchargement impossible ({url}) : {exc}") from exc
    if len(body) > MAX_BYTES:
        raise SourceError("Fichier distant trop volumineux.")
    return parse_football_data_csv(body, competition=LEAGUES[league] + " " + season)
