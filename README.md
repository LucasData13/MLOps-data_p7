# Ce dossier contient l'ensemble du code élaboré dans le cadre du projet n°7 de la formation Data Scientist OpenClassrooms "Implétmentez un modèle de scoring".
# Scoring

## Projet

### Contexte

Une société financière nommée "Prêt à dépenser" propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

<div align="center">

![](/images/logo_company.png)
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
scoring
├─ images
|   └─ logo_company.png
├─ packages
|   ├─ ressources
|   |     ├─  app_encoded.csv
|   |     ├─  app_no_encoded[...].csv
|   |     └─  banking_model.pkl
|   ├─ templates
|   |     ├─  error.html
|   |     └─  index.html
|   ├─ __init__.py
|   ├─ dashboard.py
|   ├─ functions.py
|   └─ routes.py
├─ preprocess_model.ipynb
├─ Procfile
├─ README.md
├─ requirements.txt
└─ wsgi.py
```

### Environnement

- Le prétraitement des données et la modélisation  sont effectués dans le notebook `preprocess_model.ipynb` (créé et travaillé sur *Google Collaboratory*).
- Le fichier `ressources/banking_model.pkl` provient de la modélisation faite en amont dans le notebook `preprocess_model.ipynb`.
- Les fichiers `.csv` dans `ressources/` sont des données permettant une base d'informations pour le déploiement de l'application.
- Les différents outils qui nécessitent des versions précises sont indiqués dans le fichier `requirements.txt`.
- Le projet a entièrement été développé, testé et déployé sous *Linux*. Cela risque des comportements différents sous un environnement *MacOS* ou *Windows*.
- Cette application est déployé avec *Heroku* à l'adresse suivante : [http://nh-p7-scoring.herokuapp.com/](https://nh-p7-scoring.herokuapp.com/)


### Utilisation

Une note méthodologique pour le pré-traitement et la modélisation a été rédigée mais non publiée dans ce repo.  
En ce qui concerne l'exécution du projet, vous pouvez le télécharger et le déployer en local en exécutant le fichier `wsgi.py` (à condition d'avoir bien évidemment, tous les packages nécessaires dans un environnement *Linux*).
