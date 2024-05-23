# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:08:45 2024

@author: Utilisateur
"""

import pytest
import requests
#from fastapi.testclient import TestClient
#from app.main import app, ClientData
from stream_mod.stream_module import DataClients

#BASE_URL = "http://localhost:8000/predict"
BASE_URL = 'https://apigamba-6f486e3c76df.herokuapp.com/predict'

#client = TestClient(app)

def test_predict():
    # Exemple de données de test
    test_data = {'features': 
     {str(key): 0.0 for key in DataClients.LoadData()[0].columns}
     }

    #response = client.post("/predict", json=test_data)
    response = requests.post(BASE_URL, json=test_data)
    assert response.status_code == 200, "L'API n'est pas accessible"
    json_response = response.json()
    assert "prediction" in json_response
    assert "probability" in json_response
    assert isinstance(json_response["prediction"], int), 'la prédiction doit être de type int'
    assert isinstance(json_response["probability"], float), 'la probabilié doit être de type float'

if __name__ == "__app__":
    pytest.main()
