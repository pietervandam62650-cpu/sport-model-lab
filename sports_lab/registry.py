"""Registre de prévisions (SQLite) : enregistrements figés, hachés, corrigés par
nouvel enregistrement ; résultats ajoutés une seule fois ; export/import JSON.

Limites : les triggers SQLite et les hachages détectent les modifications
accidentelles ou naïves, pas un attaquant ayant accès au fichier (qui peut
supprimer les triggers et recalculer les hachages). Pour une preuve d'antériorité,
exporter régulièrement le JSON et l'horodater ailleurs.
"""
from __future__ import annotations

import hashlib
import io
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .config import db_path
from .data import validate_results
from .evaluation import metrics, sample_warning
from .models import outcome_of

MARKETS = {"1X2": True, "ML": False}  # marché -> nul possible ?
HASH_FIELDS = ["id", "created_at", "backfill", "sport", "competition", "event_date", "home_team",
               "away_team", "market", "p_home", "p_draw", "p_away", "model_version", "model_params",
               "sources", "uncertainty", "odds_home", "odds_draw", "odds_away", "odds_time",
               "supersedes_id"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
  id TEXT PRIMARY KEY, created_at TEXT NOT NULL, backfill INTEGER NOT NULL DEFAULT 0,
  sport TEXT NOT NULL, competition TEXT NOT NULL DEFAULT '', event_date TEXT NOT NULL,
  home_team TEXT NOT NULL, away_team TEXT NOT NULL, market TEXT NOT NULL,
  p_home REAL NOT NULL, p_draw REAL NOT NULL, p_away REAL NOT NULL,
  model_version TEXT NOT NULL, model_params TEXT NOT NULL DEFAULT '{}',
  sources TEXT NOT NULL DEFAULT '', uncertainty TEXT NOT NULL DEFAULT '',
  odds_home REAL, odds_draw REAL, odds_away REAL, odds_time TEXT,
  supersedes_id TEXT, record_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
  forecast_id TEXT PRIMARY KEY REFERENCES forecasts(id), recorded_at TEXT NOT NULL,
  home_score INTEGER NOT NULL, away_score INTEGER NOT NULL, outcome TEXT NOT NULL,
  brier REAL NOT NULL, log_loss REAL NOT NULL, result_source TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS datasets (
  name TEXT PRIMARY KEY, saved_at TEXT NOT NULL, source TEXT NOT NULL DEFAULT '',
  n_rows INTEGER NOT NULL, csv TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS forecasts_no_update BEFORE UPDATE ON forecasts
  BEGIN SELECT RAISE(ABORT, 'forecasts is immutable'); END;
CREATE TRIGGER IF NOT EXISTS forecasts_no_delete BEFORE DELETE ON forecasts
  BEGIN SELECT RAISE(ABORT, 'forecasts is immutable'); END;
CREATE TRIGGER IF NOT EXISTS results_no_update BEFORE UPDATE ON results
  BEGIN SELECT RAISE(ABORT, 'results is immutable'); END;
CREATE TRIGGER IF NOT EXISTS results_no_delete BEFORE DELETE ON results
  BEGIN SELECT RAISE(ABORT, 'results is immutable'); END;
"""
TRIGGERS = ["forecasts_no_update", "forecasts_no_delete", "results_no_update", "results_no_delete"]


def compute_hash(rec: dict) -> str:
    payload = {k: rec.get(k) for k in HASH_FIELDS}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _iso_date(d) -> str:
    if isinstance(d, datetime):
        return d.date().isoformat()
    if isinstance(d, date):
        return d.isoformat()
    return date.fromisoformat(str(d)).isoformat()


class Registry:
    def __init__(self, path=None):
        self.path = Path(path) if path else db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    # ------------------------------------------------------------ prévisions
    def add_forecast(self, *, sport, home_team, away_team, event_date, probs, market="1X2",
                     model_version, competition="", model_params=None, sources="", uncertainty="",
                     odds=(None, None, None), odds_time=None, supersedes_id=None,
                     backfill=False, now=None) -> str:
        if market not in MARKETS:
            raise ValueError("Marché inconnu : " + str(market))
        has_draw = MARKETS[market]
        home_team, away_team = str(home_team).strip(), str(away_team).strip()
        if not home_team or not away_team or home_team == away_team:
            raise ValueError("Deux équipes/joueurs distincts sont requis.")
        ph, pdr, pa = (float(p) for p in probs)
        if min(ph, pdr, pa) < 0 or max(ph, pdr, pa) > 1:
            raise ValueError("Probabilités hors de [0, 1].")
        if not has_draw and pdr > 1e-9:
            raise ValueError("Marché sans nul : p_draw doit valoir 0.")
        total = ph + pdr + pa
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"La somme des probabilités vaut {total:.3f}, attendu 1 (±0,01).")
        ph, pdr, pa = ph / total, pdr / total, pa / total
        for o in odds:
            if o is not None and float(o) <= 1.0:
                raise ValueError("Une cote décimale doit être > 1.")
        created = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        ev = _iso_date(event_date)
        if not backfill and ev < created.date().isoformat():
            raise ValueError("Match déjà passé : une prévision a posteriori doit être marquée backfill=True "
                             "(elle sera exclue des évaluations).")
        with self._conn() as con:
            if supersedes_id and not con.execute("SELECT 1 FROM forecasts WHERE id=?", (supersedes_id,)).fetchone():
                raise ValueError("supersedes_id inconnu.")
            rec = {
                "id": uuid.uuid4().hex, "created_at": created.isoformat(timespec="seconds"),
                "backfill": int(bool(backfill)), "sport": str(sport).strip().casefold(),
                "competition": str(competition).strip(), "event_date": ev, "home_team": home_team,
                "away_team": away_team, "market": market, "p_home": ph, "p_draw": pdr, "p_away": pa,
                "model_version": str(model_version), "model_params": json.dumps(model_params or {}, sort_keys=True),
                "sources": str(sources), "uncertainty": str(uncertainty),
                "odds_home": odds[0], "odds_draw": odds[1], "odds_away": odds[2],
                "odds_time": odds_time, "supersedes_id": supersedes_id,
            }
            rec["record_hash"] = compute_hash(rec)
            cols = list(rec)
            con.execute(f"INSERT INTO forecasts ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                        [rec[c] for c in cols])
        return rec["id"]

    def add_result(self, forecast_id, home_score, away_score, source="", now=None) -> dict:
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        with self._conn() as con:
            f = con.execute("SELECT * FROM forecasts WHERE id=?", (forecast_id,)).fetchone()
            if not f:
                raise ValueError("Prévision inconnue.")
            if con.execute("SELECT 1 FROM results WHERE forecast_id=?", (forecast_id,)).fetchone():
                raise ValueError("Un résultat est déjà enregistré (non modifiable).")
            if now.date().isoformat() < f["event_date"]:
                raise ValueError("Le match n'a pas encore eu lieu : résultat refusé.")
            hs, as_ = int(home_score), int(away_score)
            if hs < 0 or as_ < 0:
                raise ValueError("Score négatif.")
            has_draw = MARKETS[f["market"]]
            outcome = outcome_of(hs, as_, has_draw)  # lève ValueError si nul sans nul
            row = pd.DataFrame([{"p_home": f["p_home"], "p_draw": f["p_draw"], "p_away": f["p_away"],
                                 "actual": outcome}])
            m = metrics(row, has_draw)
            con.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?,?)",
                        (forecast_id, now.isoformat(timespec="seconds"), hs, as_, outcome,
                         m["brier"], m["logloss"], str(source)))
        return {"outcome": outcome, "brier": m["brier"], "log_loss": m["logloss"]}

    def list_forecasts(self) -> pd.DataFrame:
        with self._conn() as con:
            return pd.read_sql_query(
                "SELECT f.id, f.created_at, f.backfill, f.sport, f.competition, f.event_date, f.home_team, "
                "f.away_team, f.market, f.p_home, f.p_draw, f.p_away, f.model_version, f.odds_home, "
                "f.odds_draw, f.odds_away, f.sources, f.uncertainty, f.supersedes_id, "
                "r.home_score, r.away_score, r.outcome, r.brier, r.log_loss "
                "FROM forecasts f LEFT JOIN results r ON r.forecast_id=f.id ORDER BY f.created_at DESC", con)

    def evaluate(self, since_days=None, include_backfill=False, now=None) -> pd.DataFrame:
        df = self.list_forecasts()
        df = df[df["outcome"].notna()]
        if not include_backfill:
            df = df[df["backfill"] == 0]
        if since_days is not None and len(df):
            cutoff = ((now or datetime.now(timezone.utc)) - timedelta(days=since_days)).date().isoformat()
            df = df[df["event_date"] >= cutoff]
        rows = []
        for (mv, mk), g in df.groupby(["model_version", "market"]):
            hd = MARKETS[mk]
            pred = g.rename(columns={"outcome": "actual"})
            m = metrics(pred, hd)
            uni = pred.copy()
            uni[["p_home", "p_draw", "p_away"]] = [1 / 3, 1 / 3, 1 / 3] if hd else [0.5, 0.0, 0.5]
            u = metrics(uni, hd)
            rows.append({"model_version": mv, "market": mk, "n": m["n"], "brier": m["brier"],
                         "logloss": m["logloss"], "accuracy": m["accuracy"],
                         "brier_uniforme": u["brier"], "logloss_uniforme": u["logloss"],
                         "avertissement": sample_warning(m["n"])})
        cols = ["model_version", "market", "n", "brier", "logloss", "accuracy", "brier_uniforme",
                "logloss_uniforme", "avertissement"]
        return pd.DataFrame(rows, columns=cols)

    # ------------------------------------------------------------ intégrité
    def verify_integrity(self) -> list:
        issues = []
        with self._conn() as con:
            present = {r["name"] for r in con.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
            issues += [f"Trigger d'immutabilité manquant : {t}" for t in TRIGGERS if t not in present]
            for r in con.execute("SELECT * FROM forecasts"):
                rec = dict(r)
                if compute_hash(rec) != rec["record_hash"]:
                    issues.append(f"Prévision {rec['id']} : hachage invalide (enregistrement modifié).")
            for r in con.execute("SELECT r.*, f.market FROM results r JOIN forecasts f ON f.id=r.forecast_id"):
                try:
                    if outcome_of(r["home_score"], r["away_score"], MARKETS[r["market"]]) != r["outcome"]:
                        issues.append(f"Résultat {r['forecast_id']} : issue incohérente avec le score.")
                except ValueError:
                    issues.append(f"Résultat {r['forecast_id']} : nul enregistré sur un marché sans nul.")
            for r in con.execute("SELECT forecast_id FROM results WHERE forecast_id NOT IN (SELECT id FROM forecasts)"):
                issues.append(f"Résultat orphelin : {r['forecast_id']}")
        return issues

    # ------------------------------------------------------------ export / import
    def export_json(self) -> str:
        with self._conn() as con:
            fc = [dict(r) for r in con.execute("SELECT * FROM forecasts ORDER BY created_at")]
            rs = [dict(r) for r in con.execute("SELECT * FROM results ORDER BY recorded_at")]
        return json.dumps({"schema_version": 1, "app_version": __version__,
                           "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "forecasts": fc, "results": rs}, ensure_ascii=False, indent=1)

    def import_json(self, text: str) -> dict:
        data = json.loads(text)
        stats = {"forecasts_added": 0, "forecasts_skipped": 0, "results_added": 0, "rejected": []}
        with self._conn() as con:
            for rec in data.get("forecasts", []):
                if compute_hash(rec) != rec.get("record_hash"):
                    stats["rejected"].append(f"forecast {rec.get('id')} : hachage invalide")
                    continue
                cols = list(rec)
                cur = con.execute(f"INSERT OR IGNORE INTO forecasts ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                                  [rec[c] for c in cols])
                stats["forecasts_added" if cur.rowcount else "forecasts_skipped"] += 1
            for r in data.get("results", []):
                if not con.execute("SELECT 1 FROM forecasts WHERE id=?", (r.get("forecast_id"),)).fetchone():
                    stats["rejected"].append(f"résultat {r.get('forecast_id')} : prévision absente ou rejetée")
                    continue
                cur = con.execute("INSERT OR IGNORE INTO results VALUES (?,?,?,?,?,?,?,?)",
                                  [r[c] for c in ("forecast_id", "recorded_at", "home_score", "away_score",
                                                  "outcome", "brier", "log_loss", "result_source")])
                stats["results_added"] += cur.rowcount
        return stats

    # ------------------------------------------------------------ jeux de données
    def save_dataset(self, name: str, df: pd.DataFrame, source: str = ""):
        name = name.strip()
        if not name:
            raise ValueError("Nom de jeu de données vide.")
        with self._conn() as con:
            con.execute("INSERT OR REPLACE INTO datasets VALUES (?,?,?,?,?)",
                        (name, datetime.now(timezone.utc).isoformat(timespec="seconds"), source,
                         int(len(df)), df.to_csv(index=False)))

    def list_datasets(self) -> pd.DataFrame:
        with self._conn() as con:
            return pd.read_sql_query("SELECT name, saved_at, source, n_rows FROM datasets ORDER BY saved_at DESC", con)

    def load_dataset(self, name: str):
        with self._conn() as con:
            r = con.execute("SELECT csv FROM datasets WHERE name=?", (name,)).fetchone()
        if not r:
            raise ValueError("Jeu de données inconnu.")
        return validate_results(pd.read_csv(io.StringIO(r["csv"])))
