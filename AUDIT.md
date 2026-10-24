# Audit et corrections (v0.1 → v0.2.0)

## Fichiers d'origine examinés
`app.py`, `requirements.txt`, `README.md`, `.gitignore`, `Dockerfile`, `sample_results.csv`.
Le code d'origine ne contenait pas d'erreur de syntaxe et ses fonctions Elo/métriques s'exécutaient
(vérifié sur le CSV d'exemple). Les problèmes étaient surtout de conception et de conformité au cahier des charges.

## Problèmes corrigés
| # | Problème d'origine | Correction |
|---|---|---|
| 1 | Nul imposé aux sports sans nul (p_draw 20 % → 10 % en basket) | Modèles à 2 ou 3 issues selon le sport ; métriques à 2 classes ; égalités écartées et signalées |
| 2 | Elo incohérent (attendu avec demi-points de nul × (1 − p_nul)) | Nul de Davidson : probabilités cohérentes avec la mise à jour Elo |
| 3 | Aucune prévision d'un match futur | Modèles séquentiels dont l'état final prédit un match à venir |
| 4 | Un seul modèle, aucune référence | Références uniforme et fréquences ; Elo ; Poisson en ligne (football seulement) |
| 5 | Réglage des curseurs sur le jeu de test | Rodage / validation / test final ; réglage sur la validation seulement (testé) |
| 6 | Avantage domicile identique pour tous les sports | Configuration par sport (0 en tennis) |
| 7 | Registre : champs manquants, sport pris dans le menu, probabilités manuelles étiquetées « elo », pas d'immutabilité | Schéma complet, triggers, hachage, `supersedes_id`, `backfill`, versions de modèle exactes |
| 8 | Aucune évaluation du registre | `Registry.evaluate` (Brier, log loss, référence uniforme, avertissement d'échantillon, filtre 7 jours) |
| 9 | Stockage SQLite éphémère sans issue | Chemin configurable, jeux de données enregistrés, export/import JSON vérifié par hachage |
| 10 | Accès libre en écriture | Mot de passe optionnel `APP_PASSWORD` (variable d'environnement ou `st.secrets`) |
| 11 | `use_container_width` obsolète, versions non figées, tri instable, pas de contrôle des doublons/scores | Argument retiré, bornes de versions, tri stable, validation avec comptage des lignes écartées |
| 12 | Pas de données publiques | football-data.co.uk (parseur testé sur un fichier de test, téléchargement testé avec un faux réseau) |
| 13 | Dockerfile sans volume ni port dynamique | Volume `/data`, `PORT`, utilisateur non root, healthcheck |
| 14 | Bug trouvé par mes propres tests | L'import JSON plantait si un résultat suivait une prévision rejetée : corrigé |

## Tests réellement exécutés (Python 3.12.3, pandas 3.0.2, numpy 2.4.4)
`python -m unittest discover -s tests -t .` → **48 tests, tous réussis** (~2 s).
- Validation des données, parseur football-data (dates `dd/mm/yyyy` et `dd/mm/yy`, encodage latin-1), validation d'URL, échec réseau simulé.
- Modèles : somme des probabilités = 1, taux de nul d'Elo, cohérence Poisson.
- **Absence de fuite** : modifier les résultats futurs ne change aucune prévision passée (tous les modèles) ; modifier uniquement le segment test ne change pas les paramètres choisis.
- Métriques vérifiées contre des valeurs calculées à la main ; calibration ; probabilités implicites ; bootstrap déterministe.
- Données **simulées** (graine fixe) : Elo et Poisson battent la référence en log loss avec IC95 excluant 0.
- Registre : persistance entre instances, triggers, détection de falsification (hachage, trigger supprimé), règles de résultat, export/import, jeux de données.
- `app.py` exécuté de bout en bout avec un **faux** module streamlit (chargement, évaluation, enregistrement d'une prévision, saisie de résultat, intégrité, mot de passe).
- Performance : 3 040 matchs simulés, analyse complète ≈ 1 s.

## Ce qui n'a PAS pu être testé
- **La vraie bibliothèque Streamlit n'est pas installable dans mon environnement (pas de réseau).** `app.py` n'a jamais tourné avec elle ni dans Safari. Le faux module vérifie ma logique, pas la compatibilité d'API : lance `streamlit run app.py` et signale toute erreur.
- Téléchargement réel depuis football-data.co.uk (URL et colonnes d'après ma connaissance du site ; la saison par défaut `2526` n'est pas vérifiée).
- Build Docker, déploiement, Tailscale, rendu sur iPhone.
- Versions minimales déclarées (pandas 2.0, numpy 1.26, streamlit 1.40) : non testées.

## Problèmes et limites non résolus
- **Aucune évaluation sur données réelles n'a été faite** : on ne sait pas si ces modèles ont une valeur prédictive réelle.
- Modèles simples : pas de xG, blessures, compositions, repos, météo ; Poisson indépendant (nuls sous-estimés), pas de Dixon-Coles ; Elo sans marge de victoire ni surface (tennis) ; basket/tennis non optimisés.
- Rendement théorique historique **non implémenté** (l'heure de relevé des cotes de football-data est inconnue, le résultat serait trompeur).
- Pas de PostgreSQL : SQLite + volume persistant (ou export JSON). Un seul mot de passe partagé, pas de comptes.
- Le registre est inviolable seulement contre l'accident : quelqu'un ayant le fichier peut tout recalculer. Un résultat saisi est définitif (une erreur de saisie ne se corrige pas).
- Horodatage au jour près pour le match : on ne peut pas prouver qu'une prévision précède le coup d'envoi du jour même.
- Le bootstrap suppose des matchs échangeables (approximation). Les calibrations exigent des centaines de prévisions par marché.
- L'analyse affichée est perdue au rechargement de la page (les jeux de données et le registre, eux, sont conservés).
