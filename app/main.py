# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:59:02 2024

@author: Utilisateur
"""
# 1. Library imports
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import Dict, List
import pickle
   
# pour projet 8:
'''
import io
import base64
import matplotlib.pyplot as plt
from fastapi.responses import JSONResponse
import shap
'''
     
# Configurer les variables d'environnement pour AWS S3
'''
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'https://s3.amazonaws.com'
os.environ['AWS_ACCESS_KEY_ID'] = '<your-access-key>'
os.environ['AWS_SECRET_ACCESS_KEY'] = '<your-secret-key>'
'''

app = FastAPI()

class ClientData(BaseModel):
    features: Dict[str, float] # dict[float]
    #Dict[str, float]
    #index: int  # Ajoutez cette ligne pour inclure un index

class TotalData(BaseModel):
    data: List[Dict]

def load_model():
    '''
    model_uri = "s3://<your-bucket-name>/<path-to-model>"
    model = mlflow.sklearn.load_model(model_uri)
    '''
    '''
    model_uri = 'file:///C:/Users/Utilisateur/formation_datascientist/projet_7_implementez_un_mod%C3%A8le_de_scoring/scripts/mlruns/683398248208863224/9dc1d1f54a9740688e427862f49c7e6d/artifacts/model'
    #model_uri = "models:/best_model_11_05_2@champion" 
    model = mlflow.lightgbm.load_model(model_uri)
    '''
    '''
    model_url = 'https://github.com/LucasData13/projet7/raw/main/modelisation/model/model.pkl'
    response = requests.get(model_url)

    with open('modele.pkl', 'wb') as f:
            f.write(response.content)
    with open('modele.pkl', 'rb') as f:
            model = pickle.load(f)
    '''
    model = pickle.load(open('modelisation/model/model.pkl', 'rb'))
    
    return model

model = load_model()

# pour projet 8:
@app.post("/shap_global")
async def ShapGlobal(item: TotalData):
    #data = pd.DataFrame(item.data)
    # Compute SHAP values
    #explainer = shap.TreeExplainer(model)
    #shap_values = explainer.shap_values(data)
    '''
    #shap.waterfall_plot(shap_values[0])
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, data, plot_size=[15, 9], plot_type="dot", show=False)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    #plt.close(fig)
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return JSONResponse(content={"image": img_base64})
    
    img_binary = buf.getvalue()
    
    return {"image": img_binary}
    '''

@app.post("/predict")
async def predict(client_data: ClientData):  #: ClientData
    try:
        input_data = pd.DataFrame(data= client_data.features, index=[0]) # [client_data.features]   client_data.features
        
        probability = model.predict_proba(input_data)[0][1]
        prediction = int(probability > 0.411)
        
        return {"prediction": prediction, "probability": probability}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
        
'''
# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
'''