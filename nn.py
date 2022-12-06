# Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from deep_translator import GoogleTranslator
import re 
from math import isnan
import wandb
import random
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
from xgboost import XGBClassifier
# utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_seq_items', None)

# dtype={'type': str} prevents being confused with data type for large data sets
train = pd.read_csv('data/train.csv', index_col='id', dtype={'type': str})
test = pd.read_csv('data/test.csv', index_col='id', dtype={'type': str})
train_translated = pd.read_csv('data/train_translated.csv', dtype={'type': str})
test_translated = pd.read_csv('data/test_translated.csv', index_col='id', dtype={'type': str})
combined_data = pd.read_csv('data/combined_data.csv', index_col='id', dtype={'type': str})
combined_data_translated = pd.read_csv('data/combined_data_translated.csv', index_col='id', dtype={'type': str})
combined_data_fully_translated = pd.read_csv('data/combined_data_fully_translated.csv', index_col='id', dtype={'type': str})
prep = pd.read_csv('data/prep.csv', index_col='id', dtype={'type': str})
test_prep = pd.read_csv('data/test_prepared.csv', index_col='id', dtype={'type': str})
train_prep = pd.read_csv('data/train_prepared.csv', index_col='id', dtype={'type': str})

data = train_prep.copy()

features = data.drop('type', axis=1)
labels = data.type
# at least xgboost cannot deal with string labels
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(labels)
labels = label_encoder.transform(labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=0)

from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

clf = TabNetClassifier()
clf.fit(
  X_train.values, y_train,
  eval_set=[(X_test.values, y_test)], patience=0
)

preds = clf.predict(X_test.values)
probs = clf.predict_proba(X_test.values)
val_acc = accuracy_score(y_test, preds)
val_acc

clf.save_model('models/nn/first_try')