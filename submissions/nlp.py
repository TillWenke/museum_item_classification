# Imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from deep_translator import GoogleTranslator
import re 
from math import isnan

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
text = pd.read_csv('data/text.csv', index_col='id', dtype={'type': str})
babbage = pd.read_csv('data/embedded_1k_reviews.csv', index_col='id', dtype={'type': str})
curie = pd.read_csv('data/curie.csv', dtype={'type': str})

df = pd.read_csv('data/train_curie.csv', index_col='id', dtype={'type': str})
df['curie_similarity'] = df.curie_similarity.apply(eval).apply(np.array)

from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(
    list(df.curie_similarity.values),
    df.type,
    test_size = 0.2,
    random_state=42
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
 
clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
probas = clf.predict_proba(X_test)

report = classification_report(y_test, preds)
print(report)

clf.save_model('models/curie_model.json')