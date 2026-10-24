"""Sports Model Lab : interface Streamlit (conçue pour un écran de téléphone).
La logique est dans le paquet sports_lab ; ce fichier ne fait qu'afficher."""
from __future__ import annotations

import hmac
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from sports_lab import __version__
from sports_lab.config import db_path, get_secret, sport_config
from sports_lab.data import DataError, prepare_sport, validate_results
from sports_lab.registry import MARKETS, Registry
from sports_lab.sources import LEAGUES, SourceError, fetch_football_data
from sports_lab.workflows import analyze_sport, forecast

HERE = Path(__file__).parent
st.set_page_config(page_title="Sports Model Lab", page_icon="📊", layout="centered",
                   initial_sidebar_state="collapsed")


@st.cache_resource
def get_registry() -> Registry:
    return Registry(db_path())


def gate() -> bool:
    """Mot de passe optionnel (variable d'environnement ou st.secrets APP_PASSWORD)."""
    expected = get_secret("APP_PASSWORD")
    if not expected:
        return True
    if st.session_state.get("auth"):
        return True
    typed = st.text_input("Mot de passe", type="password")
    if typed:
        if hmac.compare_digest(typed.encode(), str(expected).encode()):
            st.session_state["auth"] = True
            st.rerun()
        else:
            st.error("Mot de passe incorrect.")
    return False


def fmt(df: pd.DataFrame) -> pd.DataFrame:
    return df.round(4)


st.title("📊 Sports Model Lab")
st.caption(f"v{__version__} · évaluation statistique de modèles, pas de conseil de pari. "
           "Aucune prévision n'est une certitude.")
if not gate():
    st.stop()
reg = get_registry()
if not get_secret("APP_PASSWORD"):
    st.warning("Aucun mot de passe configuré (APP_PASSWORD) : toute personne ayant l'adresse peut écrire dans le registre.")

tab_data, tab_eval, tab_pred, tab_reg = st.tabs(["Données", "Évaluation", "Prévision", "Registre"])

# ------------------------------------------------------------------ Données
with tab_data:
    source = st.radio("Source", ["Fichier CSV", "football-data.co.uk", "Exemple (démo)", "Jeu enregistré"])
    raw, label = None, ""
    if source == "Fichier CSV":
        up = st.file_uploader("CSV : sport, event_date, home_team, away_team, home_score, away_score", type=["csv"])
        if up is not None and st.button("Charger ce fichier"):
            try:
                raw, label = pd.read_csv(up), f"CSV importé ({up.name})"
            except Exception as exc:
                st.error(f"Lecture impossible : {exc}")
    elif source == "football-data.co.uk":
        st.caption("Source publique, football uniquement. Cotes : heure de relevé inconnue. "
                   "Vérifier les conditions d'utilisation du site.")
        league = st.selectbox("Championnat", list(LEAGUES), format_func=lambda c: LEAGUES[c])
        season = st.text_input("Saison (2425 = 2024/2025)", value="2526")
        if st.button("Télécharger"):
            try:
                raw, label = fetch_football_data(season, league), f"football-data.co.uk {league} {season}"
            except (SourceError, DataError) as exc:
                st.error(str(exc))
    elif source == "Exemple (démo)":
        st.caption("Données fictives : elles illustrent le format et ne permettent PAS d'évaluer un modèle.")
        if st.button("Charger l'exemple"):
            raw, label = pd.read_csv(HERE / "sample_results.csv"), "Exemple fictif"
    else:
        saved = reg.list_datasets()
        if saved.empty:
            st.info("Aucun jeu enregistré.")
        else:
            pick = st.selectbox("Jeu", saved["name"].tolist())
            if st.button("Charger"):
                try:
                    clean, rep = reg.load_dataset(pick)
                    st.session_state.update(data=clean, report=rep, label=f"Jeu enregistré : {pick}")
                except Exception as exc:
                    st.error(str(exc))
    if raw is not None:
        try:
            clean, rep = validate_results(raw)
            st.session_state.update(data=clean, report=rep, label=label, analysis={})
        except DataError as exc:
            st.error(str(exc))
    if "data" in st.session_state:
        rep = st.session_state["report"]
        st.success(f"{st.session_state['label']} : {rep['rows_out']} matchs valides sur {rep['rows_in']}.")
        if rep["dropped"]:
            st.warning("Lignes écartées : " + " ; ".join(f"{k} ({v})" for k, v in rep["dropped"].items()))
        d = st.session_state["data"]
        has_odds = bool(d[["odds_home", "odds_away"]].notna().all(axis=1).any())
        st.caption(f"Sports : {', '.join(sorted(d['sport'].unique()))} · cotes présentes : {'oui' if has_odds else 'non'}")
        name = st.text_input("Enregistrer ce jeu sous le nom", value="")
        if st.button("Enregistrer le jeu dans la base") and name.strip():
            reg.save_dataset(name, d, st.session_state["label"])
            st.success("Jeu enregistré (rechargeable après redémarrage).")

# ------------------------------------------------------------------ Évaluation
with tab_eval:
    if "data" not in st.session_state:
        st.info("Charge d'abord des données.")
    else:
        df = st.session_state["data"]
        sport = st.selectbox("Sport", sorted(df["sport"].unique()))
        st.caption("Rodage 40 % → validation 30 % (réglage des paramètres) → test final 30 % (jamais utilisé pour régler).")
        if st.button("Lancer l'évaluation"):
            sd, has_draw, notes = prepare_sport(df, sport)
            res = analyze_sport(sd, sport, has_draw)
            res["notes"] = notes
            st.session_state.setdefault("analysis", {})[sport] = res
        res = st.session_state.get("analysis", {}).get(sport)
        if res:
            if "error" in res:
                st.error(res["error"])
            else:
                for w in res["notes"] + res["warnings"]:
                    st.warning(w)
                st.subheader("Test final")
                st.caption("Brier et log loss : plus bas = mieux. d_logloss < 0 : meilleur que la référence (fréquences passées).")
                st.dataframe(fmt(res["test"]), hide_index=True)
                if res["bookmaker"] is not None:
                    st.subheader("Face aux cotes (marge retirée)")
                    st.caption("Matchs de test avec cotes complètes. Heure de relevé des cotes inconnue.")
                    st.dataframe(fmt(res["bookmaker"]), hide_index=True)
                with st.expander("Validation et réglages"):
                    st.dataframe(fmt(res["validation"]), hide_index=True)
                    if res["tuning"] is not None:
                        st.dataframe(fmt(res["tuning"]), hide_index=True)
                for name, cal in res["calibration"].items():
                    with st.expander(f"Calibration : {name}"):
                        st.caption("Tranches avec n < 20 : non fiables.")
                        st.dataframe(fmt(cal), hide_index=True)

# ------------------------------------------------------------------ Prévision
with tab_pred:
    st.caption("Enregistre une prévision AVANT le match. Elle ne pourra plus être modifiée.")
    sport_p = None
    if "data" in st.session_state:
        sport_p = st.selectbox("Sport", sorted(st.session_state["data"]["sport"].unique()), key="sport_pred")
    res = st.session_state.get("analysis", {}).get(sport_p) if sport_p else None
    manual = st.checkbox("Saisie manuelle des probabilités", value=res is None or "error" in (res or {}))
    market, probs, version, params, teams = "1X2", None, "manuel", {}, []
    home = away = ""
    if not manual and res and "error" not in res:
        mname = st.selectbox("Modèle", [m for m in res["models"] if m != "baseline_uniforme"], index=1)
        model = res["models"][mname]
        teams = sorted(model.teams)
        c1, c2 = st.columns(2)
        home = c1.selectbox("Domicile", teams)
        away = c2.selectbox("Extérieur", teams, index=min(1, len(teams) - 1))
        market = "1X2" if res["has_draw"] else "ML"
        if home == away:
            st.error("Choisis deux équipes différentes.")
        else:
            f = forecast(model, home, away)
            probs = (f["p_home"], f["p_draw"], f["p_away"])
            k1, k2, k3 = st.columns(3)
            k1.metric("Domicile", f"{f['p_home']:.0%}")
            k2.metric("Nul", f"{f['p_draw']:.0%}" if res["has_draw"] else "—")
            k3.metric("Extérieur", f"{f['p_away']:.0%}")
            if "extras" in f:
                st.caption(f"+2,5 buts : {f['extras']['over_2_5']:.0%} · deux équipes marquent : {f['extras']['btts']:.0%}")
            version = f"{mname}-v{__version__}"
            params = {**model.params(), "n_matchs_appris": len(res["preds"][mname])}
            st.caption("Estimation du modèle, avec les limites affichées dans l'onglet Évaluation.")
    with st.form("save_forecast"):
        if manual:
            c1, c2 = st.columns(2)
            home = c1.text_input("Domicile / côté A")
            away = c2.text_input("Extérieur / côté B")
            mk = st.selectbox("Marché", list(MARKETS), format_func=lambda m: {"1X2": "1X2 (avec nul)", "ML": "Vainqueur (sans nul)"}[m])
            market = mk
            p1, p2, p3 = st.columns(3)
            ph = p1.number_input("P dom.", 0.0, 1.0, 0.45, 0.01)
            pdr = p2.number_input("P nul", 0.0, 1.0, 0.25 if mk == "1X2" else 0.0, 0.01)
            pa = p3.number_input("P ext.", 0.0, 1.0, 0.30 if mk == "1X2" else 0.55, 0.01)
            probs = (ph, pdr, pa)
        sport_in = sport_p or st.text_input("Sport", value="football")
        ev = st.date_input("Date du match", value=date.today(), min_value=date.today())
        comp = st.text_input("Compétition")
        sources = st.text_area("Sources consultées (réellement)", value=st.session_state.get("label", ""))
        unc = st.text_area("Incertitudes / facteurs qui invalideraient l'estimation")
        o1, o2, o3 = st.columns(3)
        oh = o1.number_input("Cote dom.", 0.0, 1000.0, 0.0, 0.01)
        od = o2.number_input("Cote nul", 0.0, 1000.0, 0.0, 0.01)
        oa = o3.number_input("Cote ext.", 0.0, 1000.0, 0.0, 0.01)
        if st.form_submit_button("Enregistrer la prévision"):
            try:
                if probs is None:
                    raise ValueError("Aucune probabilité à enregistrer.")
                odds = tuple(o if o > 1.0 else None for o in (oh, od, oa))
                fid = reg.add_forecast(
                    sport=sport_in, home_team=home, away_team=away, event_date=ev, probs=probs, market=market,
                    model_version=version, competition=comp, model_params=params, sources=sources,
                    uncertainty=unc, odds=odds,
                    odds_time=datetime.now(timezone.utc).isoformat(timespec="seconds") if any(odds) else None)
                st.success(f"Enregistrée (id {fid[:8]}).")
            except ValueError as exc:
                st.error(str(exc))

# ------------------------------------------------------------------ Registre
with tab_reg:
    allf = reg.list_forecasts()
    st.metric("Prévisions enregistrées", len(allf))
    if not allf.empty:
        pending = allf[allf["outcome"].isna()]
        with st.expander(f"Saisir un résultat ({len(pending)} en attente)"):
            if pending.empty:
                st.caption("Rien en attente.")
            else:
                opts = {f"{r.event_date} · {r.home_team} - {r.away_team} · {r.id[:6]}": r.id for r in pending.itertuples()}
                pick = st.selectbox("Prévision", list(opts))
                c1, c2 = st.columns(2)
                hs = c1.number_input("Score domicile", 0, 1000, 0)
                as_ = c2.number_input("Score extérieur", 0, 1000, 0)
                src = st.text_input("Source du résultat")
                if st.button("Enregistrer le résultat (définitif)"):
                    try:
                        out = reg.add_result(opts[pick], hs, as_, source=src)
                        st.success(f"Issue : {out['outcome']} · Brier {out['brier']:.3f} · log loss {out['log_loss']:.3f}")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        period = st.radio("Période d'évaluation", ["7 jours", "Tout"], horizontal=True)
        ev_df = reg.evaluate(since_days=7 if period == "7 jours" else None)
        st.subheader("Qualité mesurée")
        if ev_df.empty:
            st.info("Aucune prévision évaluée sur cette période.")
        else:
            st.dataframe(fmt(ev_df), hide_index=True)
        st.subheader("Historique")
        st.dataframe(fmt(allf.drop(columns=["id", "sources", "uncertainty"])), hide_index=True)
    if st.button("Vérifier l'intégrité"):
        issues = reg.verify_integrity()
        st.success("Aucun problème détecté.") if not issues else [st.error(i) for i in issues]
    st.download_button("Exporter le registre (JSON)", reg.export_json(), "registre_previsions.json", "application/json")
    up = st.file_uploader("Restaurer depuis un export JSON", type=["json"], key="restore")
    if up is not None and st.button("Importer"):
        try:
            st.json(reg.import_json(up.getvalue().decode("utf-8")))
        except Exception as exc:
            st.error(f"Import impossible : {exc}")
