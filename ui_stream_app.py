# -*- coding: utf-8 -*-
"""
Created on Thu May 16 23:30:24 2024

@author: Utilisateur
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from stream_mod.stream_module import PlotScore, DataClients
import numpy as np
import json
from PIL import Image
import io

st.title("Prédiction de Défaut de Crédit")

# charge les données et du model
data, data_id = DataClients.LoadData()
'''
# Convertir les colonnes booléennes en entiers
for col in data.select_dtypes(include=['bool']).columns:
    data[col] = data[col].astype(int)
'''    
st.text(type(data))
'''
# Créez un formulaire pour saisir les caractéristiques
with st.form(key='client_form'):

    # affichage des données
    st.dataframe(data)
    
    # sélection d'un client à prédire
    ligne = st.number_input('Sélectionner un numéro de ligne client', value=0)
    client_info = data.iloc[ligne, :]
    st.text(f'Vous avez choisi le client n° {data_id[ligne]} :')
    st.dataframe(client_info)

    submit_button = st.form_submit_button(label='Prédire')
''' 
#client_info = st.selectbox('Sélectionner une colonne pour l\'histogramme', data.columns)

        
# Créez un formulaire pour saisir les 387 caractéristiques
st.dataframe(data)

json_data = data.to_json(orient='records')
response = requests.post("http://127.0.0.1:8000/shap", json={"data": json.loads(json_data)})
st.write(f"status_code : {response.status_code}")
st.write(response)
img_data = response.image  

# Afficher l'image dans Streamlit
#img = Image.open(io.BytesIO(img_data))
st.image(io.BytesIO(img_data), caption='Graphique SHAP', use_column_width=True)

with st.form(key='client_selection'):
    # sélection d'un client à prédire
    ligne = st.number_input('Sélectionner un numéro de ligne client', value=0)
    client_info = data.iloc[ligne, :]
    select_button = st.form_submit_button(label='Sélectionner')

if select_button:
    st.text(f'Vous avez choisi le client n° {data_id[ligne]} :')
    st.dataframe(client_info)
    
with st.form(key='client_submission'):
    features = []
    for val in client_info.values:
        features.append(val)
    submit_button = st.form_submit_button(label='Prédire')

# Assurer que tous les types NumPy sont convertis en types natifs Python
def convert_to_native_type(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return 0
    return value

# Lorsque le formulaire est soumis
if submit_button:
    features = [convert_to_native_type(item) for item in features]
    #features = [float(x) if isinstance(x, bool) and x in (True, False) else x for x in features]
    #features = [float(x)*0 for x in features]
    #features = [0 if x is None else x for x in features]
    st.write(features)
    #features_natif = [int(item) if isinstance(item, np.int64) else item for item in features_natif]
    #st.write(f"features_natif : {features_natif}")
    
    client_features = {'features': {str(key): value for key, value in zip(client_info.index, features)}} #{"features": features}
    st.dataframe(pd.DataFrame(client_features))
    #st.write(f"client_data : {client_data}")
    response = requests.post("http://127.0.0.1:8000/predict", json=client_features)
    st.write(f"status_code : {response.status_code}")
    
    if response.status_code == 200:
        prediction = response.json()
        st.write(f"Prédiction: {'Accepté' if prediction['prediction'] == 0 else 'Refusé'}")
        st.write(f"Probabilité de défaut: {prediction['probability']:.2f}")
        st.plotly_chart(PlotScore.jauge_bar(prediction['probability']))
    else:
        st.write("Erreur lors de la prédiction.")
        st.write(response.json())
    