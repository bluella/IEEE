#!/usr/bin/env python3
"""Module with all data manipulations
   P.S. Can be converted to .ipynb"""

# %%
# All the imports
import warnings
import multiprocessing
from functools import partial
from sklearn.metrics import precision_score, recall_score, confusion_matrix,\
    accuracy_score, roc_auc_score, f1_score,\
    roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from bayes_opt import BayesianOptimization
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.lib import helpers as hlp

# %%
# load prepared data
target = 'isFraud'
train = pd.read_csv(f'{hlp.DATASETS_DEV_PATH}train9.csv', index_col='index')
test = pd.read_csv(f'{hlp.DATASETS_DEV_PATH}test9.csv', index_col='index')
sub = pd.read_csv(f'{hlp.DATASETS_PRED_PATH}sample_submission.csv')

# %%
print(train.head(5))
print(test.head(5))
# %%
# train preparation

# EXPERIMENT, DELETE!!!!!!!!!!!!!!!!!!!!
###############################################
# X = train.sort_values('TransactionDT').drop(
#     [target], axis=1)
# Y = train.sort_values('TransactionDT')[target].astype(float)
# X_test = test.sort_values('TransactionDT')
# del train
# test = test[["TransactionDT", 'TransactionID']]
###############################################

X = train.sort_values('TransactionDT').drop(
    [target, 'TransactionDT', 'TransactionID'], axis=1)
Y = train.sort_values('TransactionDT')[target].astype(float)
X_test = test.sort_values('TransactionDT').drop(
    ['TransactionDT', 'TransactionID'], axis=1)
del train
test = test[["TransactionDT", 'TransactionID']]

# %%
# check
# print(train.shape, test.shape)
# print(X.shape, Y.shape, X_test.shape)
# print(X.head(5))
# print(Y.head(5))
# %%
# Cleaning infinite values to NaN
X = hlp.clean_inf_nan(X)
X_test = hlp.clean_inf_nan(X_test)

# %%
# Fill NaNs
X.fillna(-999, inplace=True)
X_test.fillna(-999, inplace=True)


# %%
# train params
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True)
# folds = TimeSeriesSplit(n_splits=n_fold)
# folds = StratifiedKFold(n_splits=n_fold, shuffle=True)

# %%
# train lgb

params1 = {'num_leaves': 256,
           'min_child_samples': 79,
           'objective': 'binary',
           'max_depth': 13,
           'learning_rate': 0.03,
           'boosting_type': 'gbdt',
           'subsample_freq': 3,
           'subsample': 0.9,
           #   'bagging_seed': 11,
           'metric': 'auc',
           'verbosity': -1,
           'reg_alpha': 0.3,
           'reg_lambda': 0.3,
           'colsample_bytree': 0.9
           }  # test_score = 0.9393 cvm = 0.93
# TST = 0.9397 cvm = 0.9756, TransID and
# TransDT are features are stratified shuffle

# params2 = {'num_leaves': 256,
#           'min_child_samples': 79,
#           'objective': 'binary',
#           'min_split_gain': 0.0001,
#           'max_depth': 13,
#           'learning_rate': 0.03,
#           "boosting_type": "gbdt",
#           "subsample_freq": 1,
#           "subsample": 0.1,
#           "bagging_fraction": 0.1,
#           "feature_fraction": 0.1,
#           #   "bagging_seed": 11, # 'categorical_feature': cat_cols
#           "metric": 'auc',
#           "verbosity": -1,
#           'reg_alpha': 1,
#           'reg_lambda': 1,
#           'colsample_bytree': 0.9,
#           'device_type': 'gpu'
#           }  # test_score = 0.9393 cvm = 0.93
# 'categorical_feature': cat_cols
# params3 = {'num_leaves': 491,
#           'min_child_weight': 0.034,
#           'feature_fraction': 0.37,
#           'bagging_fraction': 0.42,
#           'min_data_in_leaf': 106,
#           'objective': 'binary',
#           'max_depth': -1,
#           'learning_rate': 0.007,
#           "boosting_type": "gbdt",
#           "bagging_seed": 11,
#           "metric": 'auc',
#           "verbosity": -1,
#           'reg_alpha': 0.4,
#           'reg_lambda': 0.65,
#           'random_state': 47,
#           'device_type': 'gpu'
#          } # test = 0.943 with not mine features, not special 0.937 IRL


result_dict_lgb = hlp.train_model_classification(X=X,
                                                 X_test=X_test, y=Y,
                                                 params=params1, folds=folds, model_type='lgb',
                                                 eval_metric='auc', plot_feature_importance=True,
                                                 verbose=500, early_stopping_rounds=200,
                                                 n_estimators=1500, averaging='usual', n_jobs=7)


# %%
# saving results
test = test.sort_values('TransactionDT')
test['prediction_lgb'] = result_dict_lgb['prediction']
# # in case of blendingo + result_dict_xgb['prediction']
sub[target] = pd.merge(sub, test, on='TransactionID')['prediction_lgb']
sub.to_csv(f'{hlp.DATASETS_PRED_PATH}submission.csv', index=False)


# %%
# blending
# blend_base = pd.read_csv(f'{hlp.DATASETS_PRED_PATH}submission_TST_09393_CVM_09297.csv')
# print(blend_base.head())
# print(blend_base.describe())

# sub[target] = (sub[target] + blend_base[target])/2
# sub.to_csv(f'{hlp.DATASETS_PRED_PATH}submission.csv', index=False)


# %%
# get most important features
# print(result_dict_lgb['feature_importance'].groupby('feature').mean(
# ).sort_values('importance', ascending=False).head(50)['importance'].index)
