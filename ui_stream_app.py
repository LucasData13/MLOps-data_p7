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

data, data_id = DataClients.LoadData()
  
st.dataframe(data)


with st.form(key='client_selection'):
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

def convert_to_native_type(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return 0
    return value

if submit_button:
    features = [convert_to_native_type(item) for item in features]
    
    client_features = {'features': {str(key): value for key, value in zip(client_info.index, features)}} #{"features": features}
    response = requests.post("http://127.0.0.1:8000/predict", json=client_features)
    
    if response.status_code == 200:
        prediction = response.json()
        st.write(f"Prédiction: {'Accepté' if prediction['prediction'] == 0 else 'Refusé'}")
        st.write(f"Probabilité de défaut: {prediction['probability']:.2f}")
        st.plotly_chart(PlotScore.jauge_bar(prediction['probability']))
    else:
        st.write("Erreur lors de la prédiction.")
        st.write(response.json())
    