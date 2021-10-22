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

# # Inductive Pagerank [DEMO]

# Dataset source: https://www.kaggle.com/ranjeetshrivastav/fraud-detection-dataset

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

from fucc.pagerank import get_pagerank_suspicion_scores, inductive_step, splitDataFrameIntoSmaller, postprocessing_historical_edges
from fucc.utils import get_filename, export_suspicion_scores, import_suspicion_scores, multiprocessing
from tqdm import tqdm
from functools import partial
import datetime
import pickle

# Parameters
weighted = True
WORKERS = 12
lambdas = {'ST': 0.03, 'MT': 0.004, 'LT': 0.0001}
#lambdas = {'LT': 0.0001}
chunksize = 5000

output_path = ''

# ## Load Data

df = pd.read_json('/Users/raf/Dropbox/DOC/data/fraud_datasets/archive/transactions/transactions.txt',  lines=True, convert_dates=[4])

df.iloc[:, 4] = pd.to_datetime(df.iloc[:, 4])

df = df.sort_values('transactionDateTime')
df.loc[:, 'TX_ID'] = range(df.shape[0])

df = df.rename(columns={"merchantName":"TERM_MIDUID", "customerId":"CARD_PAN_ID", "isFraud": "TX_FRAUD", "transactionDateTime":"TX_DATETIME"})

df_train = df.iloc[:320000]
df_val = df.iloc[320000:400000]
df_test = df.iloc[400000:500000]

personalization_nodes = list(df_train.index)

# ## Pagerank

# +
# %%time

dict_suspicion_scores = {}
dict_G = {}

for t, lambd in lambdas.items():
    suspicion_scores, G = get_pagerank_suspicion_scores(
                              df_train,
                              t=t,
                              lambd=lambd,
                              alpha=0.000085,
                              n_jobs=WORKERS,
                              personalization_nodes=personalization_nodes,
                              weighted=weighted)


    dict_suspicion_scores[t] = suspicion_scores
    dict_G[t] = G

# +
# %%time
## Get suspicion scores for the validation part of the training data
data = df_val.set_index('TX_ID')
data = data.loc[:, ['CARD_PAN_ID', 'TERM_MIDUID']]

# We only need these columns from historical data
historical_data = df_train.set_index('TX_ID')
historical_data = historical_data.loc[:, ['CARD_PAN_ID', 'TERM_MIDUID']]

# Inductive val set processing

dict_results = {}
for t,lamb in lambdas.items():

    suspicion_scores, G = dict_suspicion_scores[t], dict_G[t]

    # split df in smaller chunks
    chunks = splitDataFrameIntoSmaller(data, chunkSize=2000)
    partial_inductive_step = partial(inductive_step, historical_edges=historical_data, suspicion_scores=suspicion_scores, G=G, t=t)

    result = multiprocessing(function=partial_inductive_step, chunks=chunks)
    #f#ilename = get_filename(filename_elements=[subset_name, t])
    
    dict_results[t] = result
    #result.to_csv(os.path.join(output_path, filename + '_output_val.csv'))

df_val_pagerank = df_val.copy()
for t, lamb in lambdas.items():
    #filename = get_filename(filename_elements=[subset_name, t])
    df_pagerank = dict_results[t]
    #df_pagerank = df_pagerank.set_index('TX_ID')
    df_pagerank = df_pagerank.filter(regex='SC_*')
    df_val_pagerank = df_val_pagerank.merge(df_pagerank, left_on='TX_ID', right_index=True) 

#filename = get_filename(filename_elements=[subset_name])
#df_val_pagerank.to_csv(os.path.join(output_path, subset_name + '_val_pagerank_inductive.csv'))

# +
# %%time
# We only need these columns from data
data = df_test.set_index('TX_ID')
data = data.loc[:, ['CARD_PAN_ID', 'TERM_MIDUID']]

# We only need these columns from historical data
historical_data = df_train.set_index('TX_ID')
historical_data = historical_data.loc[:, ['CARD_PAN_ID', 'TERM_MIDUID']]

# Inductive test set processing

dict_results = {}
for t,lamb in lambdas.items():

    suspicion_scores, G = dict_suspicion_scores[t], dict_G[t]


    # split df in smaller chunks
    chunks = splitDataFrameIntoSmaller(data, chunkSize=2000)
    partial_inductive_step = partial(inductive_step, historical_edges=historical_data, suspicion_scores=suspicion_scores, G=G, t=t)

    result = multiprocessing(function=partial_inductive_step, chunks=chunks)
    dict_results[t] = result


df_test_pagerank = df_test.copy()
for t, lamb in lambdas.items():
    
    df_pagerank =  dict_results[t] 
    df_pagerank = df_pagerank.filter(regex='SC_*')
    df_test_pagerank = df_test_pagerank.merge(df_pagerank, left_on='TX_ID', right_index=True)



# +
# %%time
# We only need these columns from data
data = df_test.set_index('TX_ID')
data = data.loc[:, ['CARD_PAN_ID', 'TERM_MIDUID']]

# We only need these columns from historical data
historical_data = df_train.set_index('TX_ID')
historical_data = historical_data.loc[:, ['CARD_PAN_ID', 'TERM_MIDUID']]


# transductive train set processing
dict_results = {}
for t, lamb in lambdas.items():
    print(t)
    suspicion_scores, G = dict_suspicion_scores[t], dict_G[t]


    #Split historical dataset into smaller chunks
    chunks = splitDataFrameIntoSmaller(historical_data, chunkSize=5000)
    partial_postprocessing_historical_edges = partial(postprocessing_historical_edges, suspicion_scores=suspicion_scores, t=t)
    #partial_inductive_step = partial(inductive_step, historical_edges=historical_data, suspicion_scores=suspicion_scores, G=G, t=t)
    
    result = multiprocessing(function=partial_postprocessing_historical_edges, chunks=chunks)
    dict_results[t] = result

df_train_pagerank = df_train.copy()
# Load and join ST, MT, LT data for df_train and df_train
for t, lamb in lambdas.items():
    df_pagerank = dict_results[t]
    df_pagerank = df_pagerank.filter(regex='SC_*')
    df_train_pagerank = df_train_pagerank.merge(df_pagerank, left_on='TX_ID', right_index=True)


# -

# ## XGBoost Classifier

df_train = df_train_pagerank
df_val = df_val_pagerank
df_test = df_test_pagerank

pagerank_features = list(df_train.filter(regex='SC').columns)

# +
X_train = df_train[pagerank_features]
X_val = df_val[pagerank_features]
y_train = df_train.TX_FRAUD
y_val = df_val.TX_FRAUD

X_test = df_test[pagerank_features]
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


