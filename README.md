# Sports Model Lab

Prototype d'analyse statistique multi-sports en simulation. Il n'émet pas de recommandations de mises et ne garantit aucun résultat.

## Lancer en local
1. Installer Python 3.11 ou supérieur.
2. Dans le dossier du projet :
   ```bash
   python -m venv .venv
   ```
3. Activer l'environnement :
   - Windows : `.venv\Scripts\activate`
   - macOS/Linux : `source .venv/bin/activate`
4. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
5. Lancer :
   ```bash
   streamlit run app.py
   ```

## Format CSV
Colonnes obligatoires :
- `sport`
- `event_date` (date ISO, par exemple `2025-03-01`)
- `home_team`
- `away_team`
- `home_score`
- `away_score`

Chaque ligne doit représenter un match terminé. Les scores doivent être numériques. Pour le tennis, le modèle considère les scores saisis comme des scores numériques génériques; la modélisation n'est pas optimisée pour le tennis dans cette version.

## Déploiement en ligne
Le projet peut être déployé sur un hébergeur compatible Streamlit/Python en important le dossier ou le dépôt Git. Pour conserver `sports_lab.sqlite3` entre les redéploiements, utiliser un stockage persistant ou une base de données hébergée. Sans cela, les données SQLite peuvent disparaître.

Ne publie jamais de clés API dans le code. Les données sportives en direct ne sont pas configurées : il faut choisir des fournisseurs publics/légitimes et vérifier leurs conditions d'utilisation avant d'ajouter une ingestion automatisée.

## Limites connues
- Modèle Elo de démonstration, non optimisé ni validé pour chaque sport.
- La calibration du nul est une estimation historique simplifiée.
- Le découpage temporel est un test initial, pas une preuve de performance future.
- Pas d'API live intégrée.
- Les données d'exemple servent uniquement à illustrer le format; elles ne permettent pas d'évaluer le modèle.
- Pour une évaluation sérieuse, comparer plusieurs modèles avec validation chronologique, score de Brier/log-loss, calibration et échantillons suffisants.
