# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:59:02 2024

@author: Utilisateur
"""
# 1. Library imports
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.lightgbm
import os

# Configurer les variables d'environnement pour AWS S3
'''
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://s3.amazonaws.com'
os.environ['AWS_ACCESS_KEY_ID'] = '<your-access-key>'
os.environ['AWS_SECRET_ACCESS_KEY'] = '<your-secret-key>'
'''

app = FastAPI()

class ClientData(BaseModel):
    features: list[float]

def load_model():
    '''
    model_uri = "s3://<your-bucket-name>/<path-to-model>"
    model = mlflow.sklearn.load_model(model_uri)
    '''
    model_uri = "models:/best_model_11_05_2@champion" 
    model = mlflow.lightgbm.load_model(model_uri)
    return model

model = load_model()

@app.post("/predict")
async def predict(client_data: ClientData):
    try:
        input_data = pd.DataFrame([client_data.features])
        probability = model.predict_proba(input_data)[0][1]
        prediction = int(probability > 0.411)
        return {"prediction": prediction, "probability": probability}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)