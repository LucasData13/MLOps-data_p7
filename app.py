# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:59:02 2024

@author: Utilisateur
"""
import pandas as pd
import numpy as np
import mlflow.lightgbm
import mlflow
import pickle
from flask import Flask, request, jsonify

'''
# Charger le modèle de prédiction
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
'''
# import du modèle local
logged_model = 'runs:/9dc1d1f54a9740688e427862f49c7e6d/model'
model = mlflow.lightgbm.load_model(logged_model)

# Définir le seuil optimisé (exemple)
optimal_threshold = 0.411


# création de l'API Flask-------------------------
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données envoyées en POST sous forme de JSON
    data = request.get_json()
    
    # Assurez-vous que les données sont au format attendu
    try:
        # Extraire les caractéristiques (features) des données
        features = np.array(data['features']).reshape(1, -1)
    except KeyError:
        return jsonify({'error': 'Les données doivent contenir une clé "features".'}), 400
    
    # Faire la prédiction de la probabilité
    probability_of_default = model.predict_proba(features)[0, 1]
    
    # Déterminer la classe en fonction du seuil optimisé
    if probability_of_default >= optimal_threshold:
        decision = 'refusé'
    else:
        decision = 'accepté'
    
    # Retourner les résultats sous forme de JSON
    result = {
        'probability_of_default': probability_of_default,
        'decision': decision
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)