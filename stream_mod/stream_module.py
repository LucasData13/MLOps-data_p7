# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:04:40 2024

@author: Utilisateur
"""

import plotly.graph_objects as go
import pandas as pd
import os

class PlotScore:
        
    def jauge_bar(score: float):
        fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "Score de Probabilité"},
        gauge = {
            'axis': {'range': [0, 1]},
            'steps' : [
                {'range': [0, 0.175], 'color': "green"},
                {'range': [0.175, 0.35], 'color': "lightgreen"},
                {'range': [0.35, 0.407], 'color': "yellow"},
                {'range': [0.407, 0.414], 'color': "gray"},
                {'range': [0.414, 0.52], 'color': "yellow"},
                {'range': [0.52, 0.7], 'color': "orange"},
                {'range': [0.7, 1], 'color': "red"}],
            'threshold' : {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score},
            'bar' : {'color': "black"}}))
                
        # Ajouter une annotation de texte à côté de la barre
        fig.add_annotation(
            x=0.40,  # Position x de l'annotation
            y=0.73,  # Position y de l'annotation
            text="Valeur critique",
            showarrow=False,
            font=dict(color="gray", size=15),
            xanchor='left'
        )
        
        
        return fig


class DataClients:
    
    def LoadSHAPvalues(local=False):
        web_data = 'https://raw.githubusercontent.com/LucasData13/projet7/main/stream_mod/global_shap.csv'
        
        path_data = web_data
        if local == True: path_data = open('stream_mod/global_shap.csv', 'rb')
        shap_data = pd.read_csv(path_data, encoding="ISO-8859-1") 
        
        return shap_data
    
    def ChangeDir():
        os.chdir('C:/Users/Utilisateur/formation_datascientist/projet_7_implementez_un_modèle_de_scoring/scripts')
    
    def LoadData(local=False):
        web_data = 'https://raw.githubusercontent.com/LucasData13/projet7/main/tests/api_data.csv'
        
        path_data = web_data
        if local == True: path_data = open('tests/api_data.csv', 'rb')
        api_data = pd.read_csv(path_data)
        
        X = api_data.drop(['SK_ID_CURR', 'TARGET', 'Unnamed: 0'], axis=1)
        X_id = api_data['SK_ID_CURR']

        X = X.fillna(0.0)
        
        return X, X_id
    
    def LoadData2(local=False):
        web_data = 'https://raw.githubusercontent.com/LucasData13/projet7/main/tests/api_data.csv'
        
        path_data = web_data
        if local == True: path_data = open('tests/api_data.csv', 'rb')
        api_data = pd.read_csv(path_data)
        
        X = api_data.drop(['TARGET', 'Unnamed: 0'], axis=1)
        X = X.fillna(0.0)
        
        return X

    def LoadApplicationData(local=False):
        web_data = 'https://raw.githubusercontent.com/LucasData13/projet7/main/tests/application_train.csv'
        
        path_data = web_data
        if local == True: path_data = open('tests/application_train.csv', 'rb')
        application_train = pd.read_csv(path_data) 
        
        return application_train
    
    def LoadDescriptionData(local=False):
        web_data = 'https://raw.githubusercontent.com/LucasData13/projet7/main/tests/HomeCredit_columns_description.csv'
        
        path_data = web_data
        if local == True: path_data = open('tests/HomeCredit_columns_description.csv', 'rb')
        desc_data = pd.read_csv(path_data, encoding="ISO-8859-1") 
        
        return desc_data
    
   
