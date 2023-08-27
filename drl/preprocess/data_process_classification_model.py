#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from functools import reduce
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
  # plot_confusion_matrix, 
  ConfusionMatrixDisplay, confusion_matrix, 
                             recall_score, precision_score, roc_auc_score, accuracy_score, 
                             log_loss, roc_curve, auc, average_precision_score, matthews_corrcoef,
                            precision_recall_curve)

from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve

numerical_orig = ['bw', 'margin%','checkin_month']

categorical_orig = [
                'hotel_id'
                , 'room_type'
                , 'rate_plan'
                , 'rateplan_type'
                , 'ab_test_bucket'
                , 'dow'
                , 'TAAP_TRACKINGCODE'
                , 'property_typ_name'
                , 'property_parnt_chain_name'
                , 'property_city_name'
                , 'property_cntry_code'
                   ]

label = ['is_booked']

df_raw = pd.read_csv('price_train.csv')
df_clean = df_raw[numerical_orig + categorical_orig + label].sample(frac = 1)
X = df_clean.drop(columns=label)
y = df_clean['is_booked']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3
                                                    , random_state=42
                                                    , stratify=y
                                                   )

print(X_train.shape)
print(X_test.shape)

numeric_features = numerical_orig
numeric_transformer = Pipeline(
    steps=[
           ("imputer", SimpleImputer(strategy='mean')),
           ("scaler", RobustScaler())]
)

categorical_features = categorical_orig
categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

Model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            # ('sampling', sampler),
            ('RF', RandomForestClassifier(bootstrap=True, 
                                      random_state=0, 
                                      n_jobs=-1, 
                                      class_weight='balanced'))
        ]
        )

param_grid = {
        # 'imputer__strategy': ['median','mean'], 
        'RF__'+'n_estimators': [100, 150, 200],
        'RF__'+'max_leaf_nodes': [8, 12, 16],
        'RF__'+'max_samples': [0.7, 0.8, 0.9]
        }

grid_search = GridSearchCV(Model, param_grid, cv = 5, 
                scoring = 'f1',
                return_train_score=True)

grid_search.fit(X_train, y_train)
        

best_param = grid_search.best_params_
print(best_param)

final_model = grid_search.best_estimator_
model_pred = final_model.predict(X_test)
model_pred_prob = final_model.predict_proba(X_test)

target_labels = ['normal', 'abnormal']
ConfusionMatrixDisplay.from_predictions(y_test, model_pred)

tn, fp, fn, tp = confusion_matrix(y_test, model_pred).ravel()
specificity = tn / (tn + fp)
precision =  tp / (tp + fp) # precision_score(y_test, model_pred)
recall = tp / (tp + fn) # recall_score(y_test, model_pred)

f1 = 2 * (precision * recall) / (precision + recall)
balanced_acc = (recall + specificity) / 2
mcc = matthews_corrcoef(y_test, model_pred)

roc_auc = roc_auc_score(y_test, model_pred_prob[::,1])
logloss = log_loss(y_test, model_pred)
acc = accuracy_score(y_test, model_pred)    
pr_auc = average_precision_score(y_test, model_pred_prob[::,1])

print(f'RF recall: {recall}\n'
      f'RF precision: {precision}\n'
      f'RF f1: {f1}\n'
      f'RF mcc: {mcc}\n'
      f'RF balanced_acc: {balanced_acc}\n'
      ) 


features_names = X.columns
l = []
for name, score in zip(X_train.columns, final_model.steps[1][1].feature_importances_):
  l.append((name, score))
  f_imp_df = pd.DataFrame(l, columns=['name', 'score']).sort_values('score')
  f_imp_df.sort_values('score', ascending=False)
