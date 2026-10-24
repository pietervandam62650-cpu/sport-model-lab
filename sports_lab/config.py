"""Configuration : chemins, secrets (jamais dans le code) et paramètres par sport.

Les valeurs par défaut par sport sont des POINTS DE DÉPART DE DÉMONSTRATION,
pas des valeurs validées. L'onglet « Évaluation » sert justement à les tester.
"""
from __future__ import annotations

import os
from pathlib import Path

SPORT_DEFAULTS = {
    # draws : True/False/None (None = détecté d'après les données)
    "football": {"draws": True, "home_adv": 55.0, "k": 20.0, "goals_model": True},
    "basketball": {"draws": False, "home_adv": 60.0, "k": 20.0, "goals_model": False},
    # En tennis, « domicile » est arbitraire : pas d'avantage.
    "tennis": {"draws": False, "home_adv": 0.0, "k": 24.0, "goals_model": False},
}
GENERIC = {"draws": None, "home_adv": 30.0, "k": 20.0, "goals_model": False}


def sport_config(sport: str) -> dict:
    return {**GENERIC, **SPORT_DEFAULTS.get(str(sport).strip().casefold(), {})}


def db_path() -> Path:
    """Chemin de la base SQLite. Sur un hébergeur, pointer vers un disque persistant."""
    return Path(os.environ.get("SPORTS_LAB_DB", "data/sports_lab.sqlite3"))


def get_secret(name: str, default=None):
    """Lit un secret dans l'environnement, puis dans st.secrets. Jamais dans le code."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        import streamlit as st

        return st.secrets.get(name, default)
    except Exception:  # pas de fichier secrets, ou Streamlit absent
        return default
