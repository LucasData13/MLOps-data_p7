# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:06:54 2024

@author: Utilisateur
"""
#______DATA DRIFT_______________________
import pandas as pd
from evidently.report.report import Report
from evidently.metric_preset.data_drift import DataDriftPreset

# chemin d'accès aux données
path_data = 'C:\\Users\\Utilisateur\\formation_datascientist\\projet_7_implementez_un_modèle_de_scoring\\projet_7_implementez_un_modèle_de_scoring\\project_data\\'

# chargement des données de référence hypothétiques
ref = pd.read_csv(path_data + 'application_train.csv').drop('TARGET', axis=1)
# chargement des données de courantes hypothétiques
cur = pd.read_csv(path_data + 'application_test.csv')
# création de l'objet de calcul de dérive
data_drift_report = Report(metrics=[
    DataDriftPreset(),
])
# calcul des dérives entre les datasets
data_drift_report.run(reference_data=ref, current_data=cur)
# export du résultat en htlm
data_drift_report.save_html('data_drift_report.html')

drift_info = data_drift_report.as_dict()
drift_info_df = data_drift_report.as_dataframe()['DataDriftTable']

mask = drift_info_df.drift_detected == True
columns_drift = drift_info_df.loc[mask, 'column_name'].index.tolist()

#_______ANALYSE SHAP___________________________
# comparaison avec principales features_________________________________________
import mlflow.lightgbm
import shap
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# import des données traitées
path_data = "C:\\Users\\Utilisateur\\formation_datascientist\\projet_7_implementez_un_modèle_de_scoring\\"
train_complete = pd.read_csv(path_data + 'train_complete.csv')
train_complete = train_complete.drop('SK_ID_CURR', axis=1)

X_train_initial = train_complete.drop('TARGET', axis=1)
y_train_initial = train_complete.TARGET


def train_test_spliter(X, y, sampling=0.3):
    # extraction d'un échantillon de sampling % de l'ensemble
    X_train_step_1, X_extracted, y_train_step_1, y_extracted = train_test_split(
        X, y, test_size = sampling, stratify=y, random_state=0)
    # création d'un set de test
    X_train_step_2, X_test, y_train_step_2, y_test = train_test_split(
        X_extracted, y_extracted, test_size=0.2, stratify=y_extracted, random_state=0)
    # création d'un set de validation
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_step_2, y_train_step_2, test_size=0.25, stratify=y_train_step_2, random_state=0)
    return X_train, y_train, X_valid, y_valid, X_test, y_test

sampling_rate = 0.30
X_train, y_train, X_valid, y_valid, X_test, y_test = train_test_spliter(X_train_initial,
                                                                        y_train_initial,
                                                                        sampling=sampling_rate)

logged_model = 'runs:/9dc1d1f54a9740688e427862f49c7e6d/model'
best_lgb_11_05_2 = mlflow.lightgbm.load_model(logged_model)

shap.initjs()
# feature importance globale__________________________________________________________
# Create a SHAP explainer
explainer = shap.TreeExplainer(best_lgb_11_05_2)

# SHAP sur l'entraînement--------------------------------------------------------------------------------------
# Affichage des valeurs globales absolues de SHAP pour les caractéristiques les plus importantes---------------
shap_values = explainer.shap_values(X_train)

def is_impacted(feat):
    """
    indique si le feature est impacté par le data drift
    Parameters
    ----------
    feat : str
        nom de la feature transfomée.

    Returns
    -------
    bool.

    """
    include = False
    for f in columns_drift:
        if f in feat:
            include = True
            
    return include

def PlotAbsolute_GreaterSHAP(data, titre):
    #shap_values = explainer.shap_values(data)
    importance_abs_array = np.mean(np.abs(shap_values), axis=0)
    explainer_df = pd.DataFrame(data = importance_abs_array, index = data.columns, columns= ['absolute_importance_mean'])
    explainer_df_sort = explainer_df.sort_values('absolute_importance_mean', ascending=True)
    explainer_df_sort_head = explainer_df_sort.tail(25)
    
    # extraction des features data driftées
    shap_feat = explainer_df_sort_head.index.tolist() # main features
    impacted_feat = [feat for feat in shap_feat if is_impacted(feat) == True]
    not_impacted = [feat for feat in shap_feat if is_impacted(feat) == False]
    # extraction
    explainer_impacted = explainer_df_sort_head.loc[impacted_feat,:]
    explainer_not_impacted = explainer_df_sort_head.loc[not_impacted,:]
    
    # values
    vals_imp = explainer_impacted.values
    vals_imp_scalaire = [vals_imp[i][0] for i in range(len(vals_imp))]
    vals_not_imp = explainer_not_impacted.values
    vals_not_imp_scalaire = [vals_not_imp[i][0] for i in range(len(vals_not_imp))]
    
    ticks_color = (0.30, 0.2, 0.0)
    
    plt.figure(figsize=(10,6))
    plt.barh(impacted_feat, vals_imp_scalaire, color=(1.0, 0.1, 0.1), edgecolor='gray', label='Data drift detected')
    plt.barh(not_impacted, vals_not_imp_scalaire, color=(1.0, 0.5, 0.15), edgecolor='gray', label='Data drift NOT detected')
    plt.xlabel(f'mean absolute shap importance ({data.shape[0]} instances)', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title(titre, fontsize=15, fontweight='bold')
    plt.yticks(fontsize=9, color=ticks_color)
    plt.xticks(fontsize=10, color=ticks_color)
    plt.grid(axis='x')
    plt.legend()
    plt.show()

PlotAbsolute_GreaterSHAP(X_train, "Diagramme des plus grandes contributions moyennes absolues dans l'entraînement")
