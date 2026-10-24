# Utiliser l'application depuis Safari (iPhone)

**État de ce guide : rien de ce qui suit n'a été déployé ni testé** (pas d'accès réseau ni de Docker dans
l'environnement de développement). Vérifie les offres, prix et fonctions de chaque hébergeur avant de choisir.

L'application est un serveur web : l'iPhone n'exécute rien, Safari affiche la page. Il faut donc héberger
l'application quelque part, avec **un stockage persistant** pour `SPORTS_LAB_DB`, sinon le registre disparaît.

## Option A : ton ordinateur + Tailscale (sans exposition publique)
1. Sur l'ordinateur : suivre « Lancer en local » du README, puis
   `APP_PASSWORD=... streamlit run app.py --server.address=0.0.0.0`.
2. Installer Tailscale sur l'ordinateur et sur l'iPhone, même compte.
3. Dans Safari : `http://<nom-ou-ip-tailscale-de-l'ordinateur>:8501`.
Avantage : base SQLite sur ton disque, rien d'exposé sur Internet. Limite : l'ordinateur doit rester allumé.

## Option B : hébergeur Docker avec disque/volume persistant
1. Déposer le dossier dans un dépôt Git privé (le `Dockerfile` est fourni).
2. Créer un service web depuis le Dockerfile ; **monter un volume sur `/data`**.
3. Variables d'environnement : `APP_PASSWORD` (obligatoire en ligne), `SPORTS_LAB_DB=/data/sports_lab.sqlite3`.
4. Le service doit écouter sur la variable `PORT` fournie par l'hébergeur (le Dockerfile la gère, 8501 par défaut).
5. Ouvrir l'URL HTTPS fournie dans Safari.
Rappel : un seul serveur et un seul volume (SQLite n'est pas fait pour plusieurs instances en parallèle).

## Option C : Streamlit Community Cloud (la plus simple, stockage non fiable)
À ma connaissance le disque n'y est pas garanti persistant : la base peut être réinitialisée au redémarrage.
Si tu l'utilises, **exporte le registre en JSON après chaque session** (onglet Registre) et restaure-le au besoin.
Définir `APP_PASSWORD` dans les « Secrets » de l'application.

## Ajouter à l'écran d'accueil
Safari → bouton Partager → « Sur l'écran d'accueil ».

## Clés d'API et secrets
La version actuelle n'a besoin d'aucune clé. Pour tout fournisseur futur : variable d'environnement ou
`.streamlit/secrets.toml` (ignoré par git), lue avec `sports_lab.config.get_secret("NOM")`. Ne jamais mettre
une clé dans le code ni dans un dépôt.

## Avant la mise en ligne
- `APP_PASSWORD` défini (sinon n'importe qui peut écrire dans le registre).
- Une sauvegarde JSON du registre faite et testée en restauration.
- `python -m unittest discover -s tests -t .` passe sur la version déployée.
