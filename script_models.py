# -*- coding: utf-8 -*-
# pandas and numpy for data manipulation
import pandas as pd
import numpy as np

# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 22
import seaborn as sns

# Suppress warnings from pandas
import warnings
warnings.filterwarnings('ignore')

# Memory management
import gc

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import make_scorer

from sklearn.metrics import f1_score, confusion_matrix, classification_report, recall_score
from sklearn.model_selection import learning_curve

# import des données traitées
train_complete = pd.read_csv('train_complete.csv')
X_train_initial = train_complete.drop('TARGET', axis=1)
y_train_initial = train_complete.TARGET

# deux imputeurs de valeurs manquantes
imputer_mean = SimpleImputer(strategy='mean')
imputer_knn = KNNImputer(n_neighbors=5)

# équilibrage des données
smote = SMOTE(random_state=0)

# les modèles à explorer
model_logistic_1 = Pipeline([('imputer', imputer_mean), ('scaler', StandardScaler()), ('smote', smote), ('classifier', LogisticRegression())])
model_logistic_2 = Pipeline([('imputer', imputer_knn), ('scaler', StandardScaler()), ('smote', smote), ('classifier', LogisticRegression())])
model_rdmfst_1 = Pipeline([('imputer', imputer_mean), ('smote', smote), ('classifier', RandomForestClassifier(random_state=0))])
model_rdmfst_2 = Pipeline([('imputer', imputer_knn), ('smote', smote), ('classifier', RandomForestClassifier(random_state=0))])
model_xgb_1 = Pipeline([('imputer', imputer_mean), ('smote', smote), ('classifier', XGBClassifier(random_state=0))])
model_xgb_2 = Pipeline([('smote', smote), ('classifier', XGBClassifier(random_state=0))])
model_lgb_1 = Pipeline([('imputer', imputer_mean), ('smote', smote), ('classifier', LGBMClassifier(random_state=0))])
model_lgb_2 = Pipeline([('smote', smote), ('classifier', LGBMClassifier(random_state=0))])

dict_of_models_essai = {
    'model_logistic_1' : model_logistic_1,
    'model_rdmfst_1' : model_rdmfst_1,
    'model_xgb_1' : model_xgb_1,
    'model_lgb_1' : model_lgb_1
}

dict_of_models = {
    'model_logistic_1' : model_logistic_1,
    'model_logistic_2' : model_logistic_2,
    'model_rdmfst_1' : model_rdmfst_1,
    'model_rdmfst_2' : model_rdmfst_2,
    'model_xgb_1' : model_xgb_1,
    'model_xgb_2' : model_xgb_2,
    'model_lgb_1' : model_lgb_1,
    'model_lgb_2' : model_lgb_2
}

# scorer pour tracking
def score_metier(y_true, ypred):
  conf_matrix = confusion_matrix(y_true, ypred)
  TP, FN, FP, TN = conf_matrix.ravel()
  return 10 * FN + FP

scorer_metier =  make_scorer(score_metier, greater_is_better=False)

# fonction d'évaluation
def evaluation(model):

  model.fit(X_train, y_train)
  ypred_valid = model.predict(X_valid)
  ypred_train = model.predict(X_train)

  print(confusion_matrix(y_valid, ypred_valid))
  print(classification_report(y_valid, ypred_valid))
  print(score_metier(y_valid, ypred_valid))
  train_recall, valid_recall = recall_score(y_train, ypred_train), recall_score(y_valid, ypred_valid)
  print("train recall = ", train_recall)
  print("valid recall = ", valid_recall)
  print('delta train - valid = ', train_recall - valid_recall)

  #N, train_score, val_score = learning_curve(model, X_train, y_train, 
  #                                           cv=4, scoring=scorer_metier, 
  #                                           train_sizes=np.linspace(0.1, 1, 10)
  #plt.figure(figsize=(12, 8))
  #plt.plot(N, train_score.mean(axis=1), label='train score')
  #plt.plot(N, val_score.mean(axis=1), label='validation score')
  #plt.legand()
  #plt.show()
  
  
# exploration
X_train, y_train, X_valid, y_valid, X_test, y_test = train_test_spliter(X_train_initial, y_train_initial, sampling=0.001)

for name, model in dict_of_models_essai.items():
  print(name)
  evaluation(model)