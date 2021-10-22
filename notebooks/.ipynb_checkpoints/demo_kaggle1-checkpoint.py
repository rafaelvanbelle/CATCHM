# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: fucc
#     language: python
#     name: fucc
# ---

# # Demo

# ## Download data from Kaggle

# ### Install Kaggle python API

# ! pip install kaggle

# ### Authenticating with Kaggle 

# ! kaggle datasets download "ranjeetshrivastav/fraud-detection-dataset"

# ! unzip -o fraud-detection-dataset.zip

import os
import numpy as np
import pandas as pd
import networkx as nx
from nodevectors import Node2Vec
import xgboost as xgb
from fucc.inductive_step import inductive_pooling
from fucc.metrics import plot_ap, get_optimal_f1_cutoff, get_confusion_matrix
from sklearn.metrics import average_precision_score
import logging
logging.basicConfig(level=logging.INFO)

# Dataset source: https://www.kaggle.com/ranjeetshrivastav/fraud-detection-dataset

# Parameters
dimensions = 32
walk_len = 80
walk_num = 10
window_size = 5
workers = 8

# ## Load Data

df = pd.read_json('/Users/raf/Dropbox/DOC/data/fraud_datasets/archive/transactions/transactions.txt',  lines=True, convert_dates=[4])

df.iloc[:, 4] = pd.to_datetime(df.iloc[:, 4])

df = df.sort_values('transactionDateTime')
df.loc[:, 'TX_ID'] = range(df.shape[0])

df = df.rename(columns={"merchantName":"TERM_MIDUID", "customerId":"CARD_PAN_ID", "isFraud": "TX_FRAUD" })

df_train = df.iloc[:400000]
df_test = df.iloc[400000:500000]

# ## Create network

# +
G = nx.Graph()
G.add_nodes_from(df_train.TERM_MIDUID.unique(), type='merchant')
G.add_nodes_from(df_train.CARD_PAN_ID.unique(), type='cardholder')
G.add_nodes_from(df_train.TX_ID.unique(), type='transaction')

G.add_edges_from(zip(df_train.CARD_PAN_ID, df_train.TX_ID))
G.add_edges_from(zip(df_train.TX_ID, df_train.TERM_MIDUID))
# -

print(nx.info(G))

# ## Deepwalk

# +
# Fit embedding model to graph
g2v = Node2Vec(
    n_components=dimensions,
    walklen = walk_len,
    epochs = walk_num,
    w2vparams={'workers': workers, 'window': window_size}
)

g2v.fit(G)
model = g2v.model

# +
embeddings = {}
for i in df_train.TX_ID:
    embeddings[i] = model.wv[str(i)]


embeddings = pd.DataFrame().from_dict(embeddings, orient='index')


# -

df_train = df_train.merge(embeddings, left_on='TX_ID', right_index=True)

df_train.head()

# ## Inductive Pooling

results = inductive_pooling(df=df_test, embeddings=embeddings, G=G, workers=workers)

df_new_embeddings = pd.concat([pd.DataFrame(li).transpose() for li in results])

df_new_embeddings.index = df_test.TX_ID
df_test = df_test.merge(df_new_embeddings, left_on='TX_ID', right_index=True)

# ## XGBoost Classifier

embedding_features = [i for i in range(dimensions)]

# +
X_train = df_train[embedding_features].iloc[:int(df_train.shape[0]*0.8)]
X_val = df_train[embedding_features].iloc[int(df_train.shape[0]*0.8):]
y_train = df_train.TX_FRAUD.iloc[:int(df_train.shape[0]*0.8)]
y_val = df_train.TX_FRAUD.iloc[int(df_train.shape[0]*0.8):]

X_test = df_test[embedding_features]
y_test = df_test.TX_FRAUD

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)
# -

xgb_params = {
    'eval_metric': ['auc','aucpr', 'logloss'],
    'objective':'binary:logistic',
    'n_estimators': 300,
    'n_jobs':8,
    'learning_rate':0.1,
    'seed':42,
    'colsample_bytree':0.6,
    'colsample_bylevel':0.9,
    'subsample':0.9
}

model = xgb.train(xgb_params, dtrain, num_boost_round=xgb_params['n_estimators'], evals=[(dval, 'val'), (dtrain, 'train')], early_stopping_rounds=int(xgb_params['n_estimators']/2))

y_pred_proba = model.predict(dtest)


# ## Evaluation

ap = average_precision_score(y_test, y_pred_proba)
print("Average Precision: ", np.round(ap,2))

fig = plot_ap(y_test, y_pred_proba)

optimal_threshold, optimal_f1_score = get_optimal_f1_cutoff(y_test, y_pred_proba)
print("F1 Score: ", np.round(optimal_f1_score, 4))

cm = get_confusion_matrix(y_test, y_pred_proba, optimal_threshold)
print("Confusion Matrix: \n", cm)

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_pred_proba)


