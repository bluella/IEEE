#!/usr/bin/env python3
"""Find best hyperparameters for model
   P.S. Can be converted to .ipynb"""


# %%
# imports

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, TimeSeriesSplit
# Hyperopt modules
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING

from src.lib import helpers as hlp


# %%
# load prepared data
target = 'isFraud'
train = pd.read_csv(f'{hlp.DATASETS_DEV_PATH}train8.csv', index_col='index')
test = pd.read_csv(f'{hlp.DATASETS_DEV_PATH}test8.csv', index_col='index')
sub = pd.read_csv(f'{hlp.DATASETS_PRED_PATH}sample_submission.csv')

# %%
# print(train.head(5))
# print(test.head(5))
# %%
# train preparation

# EXPERIMENT, DELETE!!!!!!!!!!!!!!!!!!!!
###############################################
X = train.sort_values('TransactionDT').drop(
    [target], axis=1)
Y = train.sort_values('TransactionDT')[target].astype(float)
X_test = test.sort_values('TransactionDT')
del train
test = test[["TransactionDT", 'TransactionID']]
###############################################

# X = train.sort_values('TransactionDT').drop(
#     [target, 'TransactionDT', 'TransactionID'], axis=1)
# Y = train.sort_values('TransactionDT')[target].astype(float)
# X_test = test.sort_values('TransactionDT').drop(
#     ['TransactionDT', 'TransactionID'], axis=1)
# del train
# test = test[["TransactionDT", 'TransactionID']]

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
# optimization part
def objective(model_params):
    """Function to optimize"""
    params = {
        'max_depth': int(model_params['max_depth']),
        # 'gamma': "{:.3f}".format(model_params['gamma']),
        'subsample': int(model_params['max_depth']),
        'subsample_freq': int(model_params['subsample_freq']),
        'reg_alpha': float(model_params['reg_alpha']),
        'reg_lambda': float(model_params['reg_lambda']),
        'learning_rate': float(model_params['learning_rate']),
        'num_leaves': int(model_params['num_leaves']),
        'colsample_bytree': int(model_params['colsample_bytree']),
        'min_child_samples': int(model_params['min_child_samples']),
        'feature_fraction': float(model_params['feature_fraction']),
        'bagging_fraction': float(model_params['bagging_fraction']),
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'metric': 'auc',
        'objective': 'binary'
    }

    # print("\n############## New Run ################")
    # print(f"params = {params}")
    results_dict_lgb = hlp.train_model_classification(X=X,
                                                      X_test=X_test, y=Y,
                                                      params=params, folds=folds,
                                                      model_type='lgb',
                                                      eval_metric='auc',
                                                      plot_feature_importance=False,
                                                      verbose=500,
                                                      early_stopping_rounds=100,
                                                      n_estimators=5000,
                                                      averaging='usual',
                                                      n_jobs=6)

    print(results_dict_lgb['scores'])
    return -np.mean(results_dict_lgb['scores'])

# %%
# params for train
# train params
n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True)
# folds = TimeSeriesSplit(n_splits=n_fold)
# folds = StratifiedKFold(n_splits=n_fold, shuffle=True)

space = {
    # The maximum depth of a tree, same as GBM.
    # Used to control over-fitting as higher depth will allow model
    # to learn relations very specific to a particular sample.
    # Should be tuned using CV.
    # Typical values: 3-10
    'max_depth': hp.quniform('max_depth', 7, 23, 1),

    # reg_alpha: L1 regularization term. L1 regularization encourages sparsity
    # (meaning pulling weights to 0). It can be more useful when the objective
    # is logistic regression since you might need help with feature selection.
    'reg_alpha':  hp.uniform('reg_alpha', 0.1, 1.9),

    # reg_lambda: L2 regularization term. L2 encourages smaller weights, this
    # approach can be more useful in tree-models where zeroing
    # features might not make much sense.
    'reg_lambda': hp.uniform('reg_lambda', 0.1, 1.),

    # eta: Analogous to learning rate in GBM
    # Makes the model more robust by shrinking the weights on each step
    # Typical final values to be used: 0.01-0.2
    'learning_rate': hp.uniform('learning_rate', 0.003, 0.2),

    # colsample_bytree: Similar to max_features in GBM. Denotes the
    # fraction of columns to be randomly samples for each tree.
    # Typical values: 0.5-1
    'colsample_bytree': hp.uniform('colsample_bytree', 0.1, .9),

    # A node is split only when the resulting split gives a positive
    # reduction in the loss function. Gamma specifies the
    # minimum loss reduction required to make a split.
    # Makes the algorithm conservative. The values can vary
    # depending on the loss function and should be tuned.
    # 'gamma': hp.uniform('gamma', 0.01, .7),

    # more increases accuracy, but may lead to overfitting.
    # num_leaves: the number of leaf nodes to use. Having a large number
    # of leaves will improve accuracy, but will also lead to overfitting.
    'num_leaves': hp.choice('num_leaves', list(range(20, 500, 10))),

    # specifies the minimum samples per leaf node.
    # the minimum number of samples (data) to group into a leaf.
    # The parameter can greatly assist with overfitting: larger sample
    # sizes per leaf will reduce overfitting (but may lead to under-fitting).
    'min_child_samples': hp.choice('min_child_samples', list(range(100, 500, 10))),

    # subsample: represents a fraction of the rows (observations) to be
    # considered when building each subtree. Tianqi Chen and Carlos Guestrin
    # in their paper A Scalable Tree Boosting System recommend
    'subsample': hp.uniform('subsample', 0.1, .9),

    'subsample_freq': hp.choice('subsample_freq', list(range(0, 9, 1))),
    # randomly select a fraction of the features.
    # feature_fraction: controls the subsampling of features used
    # for training (as opposed to subsampling the actual training data in
    # the case of bagging). Smaller fractions reduce overfitting.
    'feature_fraction': hp.uniform('feature_fraction', 0.1, .9),

    # randomly bag or subsample training data.
    'bagging_fraction': hp.uniform('bagging_fraction', 0.1, .9),
    # bagging_fraction and bagging_freq: enables bagging (subsampling)
    # of the training data. Both values need to be set for bagging to be used.
    # The frequency controls how often (iteration) bagging is used. Smaller
    # fractions and frequencies reduce overfitting.
}

# Set algoritm parameters
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=30)

# Print best parameters
best_params = space_eval(space, best)
print("BEST PARAMS: ", best_params)


#%%
