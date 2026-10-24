# Sports Model Lab (v0.2.0)

Application web (Streamlit) pour **évaluer la qualité statistique** de modèles de prévision sportive,
utilisable depuis Safari sur iPhone. Elle ne donne aucun conseil de pari, ne promet aucun gain et
ne présente aucune prévision comme certaine.

## Contenu
| Fichier | Rôle |
|---|---|
| `app.py` | Interface mobile (4 onglets : Données, Évaluation, Prévision, Registre) |
| `sports_lab/data.py` | Validation des CSV (chaque ligne écartée est comptée par motif) |
| `sports_lab/sources.py` | Source publique football-data.co.uk (+ limites documentées) |
| `sports_lab/models.py` | Références (uniforme, fréquences), Elo (nul de Davidson ou 2 issues), Poisson en ligne (football) |
| `sports_lab/evaluation.py` | Walk-forward, Brier, log loss, calibration, bootstrap, comparaison aux cotes |
| `sports_lab/workflows.py` | Évaluation complète d'un sport (rodage 40 % / validation 30 % / test 30 %) |
| `sports_lab/registry.py` | Registre SQLite immuable (triggers + hachage SHA-256), export/import JSON, jeux de données |
| `tests/` | Tests reproductibles (`unittest`) |

## Lancer en local
```bash
python -m venv .venv && source .venv/bin/activate      # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
Variables facultatives : `APP_PASSWORD` (protège l'application), `SPORTS_LAB_DB` (chemin de la base,
défaut `data/sports_lab.sqlite3`). Voir `.env.example` et `.streamlit/secrets.toml.example`.

## Lancer les tests
```bash
python -m unittest discover -s tests -t . -v
```
Sans dépendance externe de test (ni pytest, ni streamlit requis : `test_app_smoke.py` utilise un faux module).

## Format CSV
Colonnes obligatoires : `sport, event_date (AAAA-MM-JJ), home_team, away_team, home_score, away_score`.
Optionnelles : `competition, odds_home, odds_draw, odds_away` (cotes décimales). Une ligne = un match terminé.
Les scores doivent être des entiers. Sans nul : basket, tennis (par défaut) ; football avec nul.

## Méthode d'évaluation
1. **Walk-forward** : chaque match est prévu avec l'état du modèle *avant* ce match, puis le modèle apprend.
2. **Trois tranches chronologiques** : rodage (40 %, non évalué) → validation (30 %, sert à choisir K et
   l'avantage domicile d'Elo) → test final (30 %, jamais utilisé pour régler).
3. **Métriques** : Brier (somme sur les issues ; uniforme à 3 issues = 0,667), log loss (uniforme = 1,099),
   exactitude, calibration par tranches (tranches < 20 matchs signalées non fiables).
4. **Référence** : fréquences passées des issues. Intervalle bootstrap à 95 % sur la différence de log loss.
   Un modèle n'est déclaré « meilleur » que si n ≥ 100 **et** l'intervalle exclut 0, et la mention
   « non garanti ensuite » est toujours affichée.
5. **Cotes** (si présentes) : probabilités implicites normalisées (marge retirée proportionnellement).

## Registre des prévisions
Chaque prévision (id, horodatage UTC, sport, compétition, marché, probabilités, version du modèle,
paramètres, sources, incertitudes, cotes et heure de relevé) est **figée** : triggers SQLite contre
UPDATE/DELETE et hachage vérifiable. Une correction = nouvel enregistrement (`supersedes_id`).
Une prévision sur un match déjà passé est refusée, sauf marquage `backfill` (exclue des évaluations).
Le résultat s'ajoute une seule fois, pas avant la date du match.

## Limites (voir AUDIT.md pour la liste complète)
Les modèles sont simples et n'utilisent ni xG, ni blessures, ni compositions. Leur bon fonctionnement
n'a été vérifié que sur des données **simulées**, pas sur des données réelles. Battre la marge d'un
bookmaker est très difficile : l'issue la plus probable de ce suivi est de constater l'absence d'avantage.
