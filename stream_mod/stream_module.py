# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:04:40 2024

@author: Utilisateur
"""

import plotly.graph_objects as go
import pandas as pd

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
    
    def LoadData():
        path_data = 'https://raw.githubusercontent.com/LucasData13/projet7/main/tests/api_data.csv'
        #path_data = "C:\\Users\\Utilisateur\\formation_datascientist\\projet_7_implementez_un_modèle_de_scoring\\scripts\\tests\\"
        api_data = pd.read_csv(path_data + 'api_data.csv')
        
        X = api_data.drop(['SK_ID_CURR', 'TARGET', 'Unnamed: 0'], axis=1)
        X_id = api_data['SK_ID_CURR']
        '''
        # Convertir les colonnes booléennes en entiers
        for col in X.columns:
            X[col] = X[col].astype(float)
        '''
        X = X.fillna(0.0)
        
        return X, X_id
