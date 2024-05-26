# Ce dossier contient l'ensemble du code élaboré dans le cadre du projet n°7 de la formation Data Scientist OpenClassrooms "Implétmentez un modèle de scoring".

### Contexte

Une société financière nommée "Prêt à dépenser" propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

<div align="center">

![](logo_company.png)
</div>

L'entreprise souhaite mettre en oeuvre un outil de "scoring crédit" pour calculer la probabilité qu'un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s'appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.)

De plus les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner.

Prêt à dépenser décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.


### Données

Les données sont disponibles sur kaggle à [cette adresse](https://www.kaggle.com/c/home-credit-default-risk/data).


### Mission

1. Construire un modèle de scoring qui donnera une prédiction sur la probabilité de faillite d'un client de façon automatique.

2. Construire un dashboard interactif à destination des gestionnaires de la relation client permettant d'interpréter les prédictions faites par le modèle, et d’améliorer la connaissance client des chargés de relation client.


## Développement

### Organisation

```
MLOps-data_p7
├─ app
|   ├─  __init__.py
|   └─  main.py
├─ modelisation
|    ├─  model
|    └─ model.pkl
|    └─  modelisation_3.py
├─ stream_mod
|   ├─  __pycache__
|   ├─  __init__.py
|   ├─  stream_module.py
|   └─  ui_stream_app.py
├─ tests
|   ├─  __init__.py
|   ├─  api_data.csv
|   └─  app_test.py
├─ .gitignore
├─ Procfile
├─ README.md
├─ data_drift_report.html
├─ evidently_datadrift.py
├─ requirements.txt
└─ logo_company.png
```

### Environnement

`modelisation/`:
- Le prétraitement des données et la modélisation  sont effectués dans le notebook `modelisation/preprocess_data.ipynb` (créé et travaillé sur *Google Collaboratory*). Ce notebook inclut tout le feature engineering pour l'amélioration des performances.
- Le fichier `modelisation/model/model.pkl` provient de la modélisation faite dans le script `modelisation/modelisation_3.py` et a fait l'objet d'une sélection parmis un grand nombre d'expériences trackées dans MLFlow.

`app/`:
- contient le script de l'API `main.py` faisant appel à `modelisation/model/model.pkl`.
`stream_mod/`:
- contient le script de l'interface d'utilisation streamlit de l'API `ui_stream_app.py` ainsi qu'un module de fonctions.

`tests/`:
- Le fichier `.csv` dans `tests/` sont des données permettant une base d'informations pour le déploiement de l'application.
- Le fichier `app_test.py` permet de réaliser des test unitaires de l'API dans le cadre du déploiement continu.

`MLOps-data_p7/` (racine):
- Les différents outils qui nécessitent des versions précises pour l'utilisation de l'API sont indiqués dans le fichier `requirements.txt`.
- `.github/` contient le fichier YAML de configuration pour l'exécution des tests unitaires automatique à chaque 'push' git dans le cadre du déploiement et intégration continus.
- Le `Procfile` contient le configuration du déploiement sur Heroku.
- `evidently_datadrift.py` a permis d'obtenir l'analyse du data-drift entre 'application_train.csv' et 'application test.csv' disponibles à l'adresse des données indiquée au début de ce document.
- `data_drift_report.html` est le résultat d'analyse du data-drift.


### API
- L'API est déployé avec *Heroku* à l'adresse suivante : [https://apigamba-6f486e3c76df.herokuapp.com/)].
- Pour visualiser la route de prédiction de l'API vous pourez vous rendre à l'adresse docs suivante : [https://apigamba-6f486e3c76df.herokuapp.com/docs#/default/predict_predict_post]


### Utilisation

Pour tester l'API, vous pouvez :
- télécharger ce répertoire en local sur votre machine
- créer un environnement virtuel
- y installer les packages listés dans `requirements.txt`
- activer l'environnement virtuel
- vous rendre dans `stream_mod/` et exécuter la commande `streamlit ui_stream_app.py`
- l'interface de test streamlit s'ouvre alors dans votre navigateur. Vous pourrez sélectionner un index de ligne de la table de données mise à disposition grâce au bouton d'incrément puis du bouton "Sélectionner"
- les caractéristiques du clients s'affichent en dessous. Vous pouvez cliquer sur le bouton "Prédire" pour faire afficher la réponse du modèle "accord" ou "refut" de crédit.

### NOTA : 
L'exécution et l'interface utilisateur ne sont pas optimisés dans le cadre de ce projet n°7 au vues des exigences demandées. Un travail plus important d'élaboration d'une interface 'Userfriendly' est réalisé dans le projet suivant.
