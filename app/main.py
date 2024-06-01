# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:59:02 2024

@author: Utilisateur
"""
# 1. Library imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import Dict
import pickle
#import uvicorn   
from shap import Explainer
#import os
   
# pour projet 8:
'''
import io
import base64
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse
import shap
'''

app = FastAPI()

class ClientData(BaseModel):
    features: Dict[str, float] # dict[float]

def load_model():
    model = pickle.load(open('modelisation/model/model.pkl', 'rb'))
    return model

model = load_model()

@app.post("/predict")
async def predict(client_data: ClientData):  #: ClientData
    try:
        input_data = pd.DataFrame(data= client_data.features, index=[0]) # [client_data.features]   client_data.features
        
        probability = model.predict_proba(input_data)[0][1]
        prediction = int(probability > 0.411)
        explainer = Explainer(model)
        shap_values = explainer.shap_values(input_data)
        
        return {"prediction": prediction, "probability": probability, "shap_values": shap_values.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
'''
# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
'''