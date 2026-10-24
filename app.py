import io, sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DB_PATH = Path("sports_lab.sqlite3")
REQUIRED = ["sport", "event_date", "home_team", "away_team", "home_score", "away_score"]

st.set_page_config(page_title="Sports Model Lab", page_icon="📊", layout="wide")

def db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL, sport TEXT NOT NULL, event_date TEXT NOT NULL,
        home_team TEXT NOT NULL, away_team TEXT NOT NULL,
        p_home REAL NOT NULL, p_draw REAL NOT NULL, p_away REAL NOT NULL,
        model_version TEXT NOT NULL, outcome TEXT, home_score REAL, away_score REAL,
        source_note TEXT DEFAULT ''
    )""")
    con.commit()
    return con

def validate_data(df):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return None, f"Colonnes manquantes : {', '.join(missing)}"
    x = df[REQUIRED].copy()
    x["event_date"] = pd.to_datetime(x["event_date"], errors="coerce", utc=True)
    x["home_score"] = pd.to_numeric(x["home_score"], errors="coerce")
    x["away_score"] = pd.to_numeric(x["away_score"], errors="coerce")
    x["sport"] = x["sport"].astype(str).str.strip()
    x["home_team"] = x["home_team"].astype(str).str.strip()
    x["away_team"] = x["away_team"].astype(str).str.strip()
    x = x.dropna(subset=["event_date","home_score","away_score"])
    x = x[(x["sport"] != "") & (x["home_team"] != "") & (x["away_team"] != "")]
    x = x[x["home_team"] != x["away_team"]]
    x = x.sort_values("event_date").reset_index(drop=True)
    return x, None

def outcome(h, a):
    return "home" if h > a else ("away" if h < a else "draw")

def softmax(vals):
    z = np.array(vals, dtype=float)
    z -= np.max(z)
    e = np.exp(z)
    return e / e.sum()

def elo_walkforward(df, k=20, home_adv=55, scale=400, initial=1500):
    """Chronological, pre-match Elo probabilities. No future results are used."""
    ratings = {}
    out = []
    for r in df.itertuples(index=False):
        sport = str(r.sport)
        hkey, akey = (sport, str(r.home_team)), (sport, str(r.away_team))
        rh, ra = ratings.get(hkey, initial), ratings.get(akey, initial)
        diff = rh + home_adv - ra
        p_home_win = 1 / (1 + 10 ** (-diff / scale))
        # Estimate draw probability only from earlier games in this sport.
        prior = [x for x in out if x["sport"] == sport]
        draw_rate = (sum(x["actual"] == "draw" for x in prior) + 2) / (len(prior) + 10)
        p_draw = min(0.35, max(0.02, draw_rate))
        p_home = p_home_win * (1 - p_draw)
        p_away = (1 - p_home_win) * (1 - p_draw)
        # Normalize and preserve draw estimate.
        s = p_home + p_draw + p_away
        probs = np.array([p_home/s, p_draw/s, p_away/s])
        actual = outcome(float(r.home_score), float(r.away_score))
        out.append({"sport": sport, "event_date": r.event_date, "home_team": str(r.home_team),
                    "away_team": str(r.away_team), "p_home": probs[0], "p_draw": probs[1],
                    "p_away": probs[2], "actual": actual,
                    "home_score": float(r.home_score), "away_score": float(r.away_score)})
        # Update ratings only after the prediction is stored.
        expected_h = 1 / (1 + 10 ** (-(rh + home_adv - ra) / scale))
        actual_h = 1.0 if r.home_score > r.away_score else (0.5 if r.home_score == r.away_score else 0.0)
        ratings[hkey] = rh + k * (actual_h - expected_h)
        ratings[akey] = ra - k * (actual_h - expected_h)
    return pd.DataFrame(out)

def metrics(pred):
    if pred.empty:
        return {}
    actual_map = {"home":0, "draw":1, "away":2}
    p = np.clip(pred[["p_home","p_draw","p_away"]].to_numpy(float), 1e-12, 1)
    y = np.array([actual_map[a] for a in pred["actual"]], dtype=int)
    onehot = np.eye(3)[y]
    brier = float(np.mean(np.sum((p-onehot)**2, axis=1)))
    logloss = float(-np.mean(np.log(p[np.arange(len(y)), y])))
    labels = np.array(["home","draw","away"])[p.argmax(axis=1)]
    return {"n":len(pred), "accuracy":float(np.mean(labels == pred["actual"])),
            "brier":brier, "logloss":logloss}

def calibration_table(pred, bins=10):
    rows=[]
    for outcome_name, col in [("Victoire domicile","p_home"),("Nul","p_draw"),("Victoire extérieur","p_away")]:
        tmp=pred[[col,"actual"]].copy()
        target = {"p_home":"home","p_draw":"draw","p_away":"away"}[col]
        tmp["observed"]=(tmp["actual"]==target).astype(float)
        tmp["bucket"]=pd.cut(tmp[col], bins=np.linspace(0,1,bins+1), include_lowest=True)
        g=tmp.groupby("bucket", observed=False).agg(prediction=(col,"mean"),observed=("observed","mean"),n=("observed","size")).reset_index()
        g["issue"]=outcome_name
        rows.append(g)
    return pd.concat(rows, ignore_index=True)

st.title("📊 Sports Model Lab")
st.caption("Laboratoire de prévision multi-sports — mode simulation, sans recommandations de mises.")
st.warning("Les probabilités sont des estimations expérimentales, pas des certitudes. Le modèle ci-dessous est un point de départ Elo, volontairement simple.")
con = db()

with st.sidebar:
    st.header("Données")
    uploaded = st.file_uploader("Importer un CSV de résultats historiques", type=["csv"])
    st.caption("Colonnes requises : sport, event_date, home_team, away_team, home_score, away_score.")
    k = st.slider("Vitesse d'ajustement Elo (K)", 5, 50, 20)
    home_adv = st.slider("Avantage domicile (points Elo)", 0, 100, 55)
    st.caption("Les paramètres sont exploratoires : comparer les modèles sur une période test chronologique.")

if uploaded:
    try:
        raw = pd.read_csv(uploaded)
        df, err = validate_data(raw)
        if err:
            st.error(err)
        elif len(df) < 20:
            st.error("Il faut idéalement au moins 20 matchs; davantage est nécessaire pour une évaluation fiable.")
        else:
            st.session_state["data"] = df
            st.success(f"{len(df)} matchs valides chargés.")
    except Exception as e:
        st.error(f"Impossible de lire le CSV : {e}")

if "data" not in st.session_state:
    st.info("Importe un CSV pour commencer. Tu peux télécharger le modèle de fichier ci-dessous.")
    sample = pd.DataFrame([
        ["football","2025-01-01","Team A","Team B",2,1],
        ["football","2025-01-03","Team C","Team A",0,0],
        ["basketball","2025-01-04","Team D","Team E",88,92],
        ["tennis","2025-01-05","Player A","Player B",2,1],
    ], columns=REQUIRED)
    st.download_button("Télécharger le modèle CSV", sample.to_csv(index=False).encode(), "modele_resultats.csv", "text/csv")
    st.stop()

df = st.session_state["data"].copy()
sports = sorted(df["sport"].unique().tolist())
selected = st.selectbox("Sport à analyser", sports)
sd = df[df["sport"] == selected].sort_values("event_date").reset_index(drop=True)
if len(sd) < 10:
    st.warning("Peu de matchs pour ce sport : les résultats seront très incertains.")

# Walk-forward evaluation: each prediction is made before the corresponding result updates ratings.
pred = elo_walkforward(sd, k=k, home_adv=home_adv)
split = max(1, int(len(pred)*0.7))
test = pred.iloc[split:].copy()
m = metrics(test)
c1,c2,c3,c4 = st.columns(4)
c1.metric("Matchs de test", m.get("n",0))
c2.metric("Exactitude", f"{m.get('accuracy',0):.1%}" if m else "—")
c3.metric("Score de Brier ↓", f"{m.get('brier',0):.3f}" if m else "—")
c4.metric("Log-loss ↓", f"{m.get('logloss',0):.3f}" if m else "—")
st.caption(f"Évaluation chronologique : premières 70 % des rencontres pour initialiser le modèle, dernières 30 % comme période de test. Évite de comparer ces chiffres entre sports sans tenir compte de leurs formats.")

st.subheader("Dernières estimations pré-match")
show = pred.tail(20).copy()
show["Date"] = show["event_date"].dt.strftime("%Y-%m-%d")
show["Prob. domicile"] = show["p_home"].map(lambda x:f"{x:.1%}")
show["Prob. nul"] = show["p_draw"].map(lambda x:f"{x:.1%}")
show["Prob. extérieur"] = show["p_away"].map(lambda x:f"{x:.1%}")
show["Résultat réel (évaluation historique)"] = show["actual"]
st.dataframe(show[["Date","home_team","away_team","Prob. domicile","Prob. nul","Prob. extérieur","Résultat réel (évaluation historique)"]], use_container_width=True, hide_index=True)

st.subheader("Calibration des probabilités — période test")
if len(test):
    cal = calibration_table(test)
    st.dataframe(cal[["issue","bucket","prediction","observed","n"]].rename(columns={
        "issue":"Issue","bucket":"Tranche de probabilité","prediction":"Probabilité moyenne prédite",
        "observed":"Fréquence observée","n":"Nombre de matchs"}), use_container_width=True, hide_index=True)
    st.caption("Un modèle bien calibré doit avoir, par exemple, environ 60 % de résultats observés dans un groupe de prévisions proches de 60 %. Les petits groupes sont très instables.")
else:
    st.info("Pas assez de matchs pour calculer la période de test.")

st.subheader("Enregistrer un pronostic de simulation")
st.caption("Cette partie permet d'enregistrer une estimation manuelle pour une rencontre future. Le résultat pourra être ajouté après le match pour l'évaluer.")
with st.form("new_prediction"):
    col1,col2 = st.columns(2)
    home = col1.text_input("Équipe/joueur à domicile", placeholder="Équipe A")
    away = col2.text_input("Équipe/joueur à l'extérieur", placeholder="Équipe B")
    event_date = st.date_input("Date du match")
    p1,p2,p3 = st.columns(3)
    ph = p1.number_input("Probabilité domicile", min_value=0.0, max_value=1.0, value=0.45, step=0.01)
    pdra = p2.number_input("Probabilité nul", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    pa = p3.number_input("Probabilité extérieur", min_value=0.0, max_value=1.0, value=0.30, step=0.01)
    note = st.text_input("Notes / sources")
    submitted = st.form_submit_button("Enregistrer en simulation")
    if submitted:
        total = ph+pdra+pa
        if not home.strip() or not away.strip() or home.strip()==away.strip():
            st.error("Indique deux équipes/joueurs différents.")
        elif abs(total-1.0)>0.01:
            st.error("La somme des probabilités doit être égale à 100 % (tolérance 1 point).")
        else:
            ph,pdra,pa = ph/total,pdra/total,pa/total
            con.execute("INSERT INTO predictions(created_at,sport,event_date,home_team,away_team,p_home,p_draw,p_away,model_version,source_note) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), selected, str(event_date), home.strip(), away.strip(), ph,pdra,pa,"elo-baseline-v1",note))
            con.commit()
            st.success("Prévision enregistrée dans la base locale du serveur.")

st.subheader("Historique enregistré")
saved = pd.read_sql_query("SELECT id,created_at,sport,event_date,home_team,away_team,p_home,p_draw,p_away,model_version,outcome,home_score,away_score,source_note FROM predictions ORDER BY id DESC", con)
if saved.empty:
    st.info("Aucune prévision manuelle enregistrée pour le moment.")
else:
    st.dataframe(saved, use_container_width=True, hide_index=True)
    st.download_button("Exporter l'historique CSV", saved.to_csv(index=False).encode(), "historique_previsions.csv", "text/csv")
    with st.expander("Saisir le résultat d'une prévision enregistrée"):
        pending = saved[saved["outcome"].isna()]
        if pending.empty:
            st.caption("Aucune prévision en attente de résultat.")
        else:
            options = {f"#{int(r.id)} — {r.sport} — {r.home_team} vs {r.away_team} ({r.event_date})": int(r.id) for r in pending.itertuples()}
            selected_label = st.selectbox("Prévision", list(options.keys()))
            rh = st.number_input("Score domicile / côté A", min_value=0.0, value=0.0, step=1.0)
            ra = st.number_input("Score extérieur / côté B", min_value=0.0, value=0.0, step=1.0)
            if st.button("Enregistrer le résultat réel"):
                actual = outcome(rh,ra)
                con.execute("UPDATE predictions SET outcome=?,home_score=?,away_score=? WHERE id=?", (actual,rh,ra,options[selected_label]))
                con.commit()
                st.success("Résultat enregistré. Recharge la page pour actualiser l'historique.")

st.caption("Version prototype : l'import CSV est la source des résultats historiques. Les connecteurs de données sportives en direct ne sont pas configurés. SQLite locale convient aux essais, mais un hébergement éphémère peut effacer la base lors d'un redéploiement.")
