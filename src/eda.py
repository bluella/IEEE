#!/usr/bin/env python3
"""Exploratory data analysis
   P.S. Can be converted to .ipynb"""

# %%
# All the imports
import multiprocessing
import warnings
import gc
from time import time
import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from tqdm import tqdm_notebook
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

from src.lib import helpers as hlp
warnings.simplefilter('ignore')
sns.set()

# %%
# Load datasets
datasets = [
    f'{hlp.DATASETS_ORIGINAL_PATH}train_identity.csv',
    f'{hlp.DATASETS_ORIGINAL_PATH}train_transaction.csv',
    f'{hlp.DATASETS_ORIGINAL_PATH}test_identity.csv',
    f'{hlp.DATASETS_ORIGINAL_PATH}test_transaction.csv',
    f'{hlp.DATASETS_PRED_PATH}sample_submission.csv']

with multiprocessing.Pool() as pool:
    train_identity, \
        train_transaction, \
        test_identity, \
        test_transaction, sub = pool.map(hlp.my_csv_read, datasets)

# Load preprocessed
# train = pd.read_csv(f'{hlp.DATASETS_DEV_PATH}train8.csv', index_col='index')
# test = pd.read_csv(f'{hlp.DATASETS_DEV_PATH}test8.csv', index_col='index')

target = 'isFraud'
# %%
# Merging transactions and identity
train = pd.merge(train_transaction, train_identity,
                 on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity,
                on='TransactionID', how='left')

# %%
# datasets shapes
print('train shape is {}'.format(train.shape))
print('test shape is {}'.format(test.shape))


# %%
# datasets heads
print(train.head())
print(test.head())


# %%
# check for missing values
missing_values_count = train.isnull().sum().sort_values(ascending=False)
print(missing_values_count.head(10))
total_cells = np.product(train.shape)
total_missing = missing_values_count.sum()
print("% of missing data = ", (total_missing/total_cells) * 100)
many_null_cols = [col for col in train.columns if
                  train[col].isnull().sum() / train.shape[0] > 0.9]
print('too many nulls in:')
print(many_null_cols)
# do the same for test
# %%
# plot column distribution
# sns.distplot(train[target], bins=[0,1])
# sns.barplot(data=train[target], estimator=lambda x: sum(x == 0)*100.0/len(x))
train[target].plot(kind='hist')
# sns.countplot(data=train[target])

# %%
# train vs test column distribution
col_to_plot = 'TransactionDT'
train[col_to_plot].plot(kind='hist',
                        figsize=(15, 5),
                        label='train',
                        bins=50,
                        title=f'Train vs Test {col_to_plot} distribution')
test[col_to_plot].plot(kind='hist',
                       label='test',
                       bins=50)
plt.legend()
plt.show()


# %%
# get categorical columns and show info
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
train_num = train.select_dtypes(include=numerics)
cat_columns = [col for col in train.columns if col not in train_num.columns]
hlp.resumetable(train[cat_columns])


# %%
# analyse specific set of columns
C_cols = []
for col in train:
    if col.startswith('C'):
        C_cols.append(col)
# print(C_cols)

# info about columns
# hlp.resumetable(train[C_cols])

# train vs test plot
# for col in C_cols:
#     col_to_plot = col
#     train[col_to_plot].plot(kind='hist',
#                             figsize=(15, 5),
#                             label='train',
#                             bins=50,
#                             title=f'Train vs Test {col_to_plot} distribution')
#     test[col_to_plot].plot(kind='hist',
#                            label='test',
#                            bins=50)
#     plt.legend()
#     plt.show()

# heatmap
# sns.heatmap(train[C_cols].corr())

# get real dimensions
# scaler = MinMaxScaler(feature_range=[0, 1])
# data_rescaled = scaler.fit_transform(train[C_cols])

# #Fitting the PCA algorithm with our Data
# pca = PCA().fit(data_rescaled)
# #Plotting the Cumulative Summation of the Explained Variance
# plt.figure()
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Variance (%)') #for each component
# plt.title('Pulsar Dataset Explained Variance')
# plt.show()

# plot feature against index or time
for col in C_cols:
    train.set_index('TransactionDT')[col].fillna(-1).plot(style='.', figsize=(15, 3))
    test.set_index('TransactionDT')[col].fillna(-1).plot(style='.', figsize=(15, 3))
    plt.show()

# %%
