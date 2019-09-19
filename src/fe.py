#!/usr/bin/env python3
"""Feature engineering
   P.S. Can be converted to .ipynb"""


# %%
# All the imports
import re
import warnings
import multiprocessing
from functools import partial
from sklearn.metrics import precision_score, recall_score, confusion_matrix,\
    accuracy_score, roc_auc_score, f1_score,\
    roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA, KernelPCA
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from src.lib import helpers as hlp
warnings.simplefilter('ignore')

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
# print(train.shape)
# print(test.shape)


# %%
# delete heavy parts
del train_identity, train_transaction, test_identity, test_transaction


# %%
# add nans count as feature

D_columns = []
for column in train.columns:
    if 'D' in column:
        D_columns.append(column)

train['D_nulls'] = train[D_columns].isnull().sum(axis=1)
test['D_nulls'] = test[D_columns].isnull().sum(axis=1)
# %%
# Lets add datetime features
train = hlp.add_datetime_info(train)
test = hlp.add_datetime_info(test)

# %%
# card features
column_card_numbers = ['card1', 'card2', 'card3', 'card5']
for column in column_card_numbers:
    train[column] = train[column].map(str).fillna(
        'NNNN').map(hlp.correct_card_id)
    test[column] = test[column].map(str).fillna(
        'NNNN').map(hlp.correct_card_id)
    for number in [1, 2, 3, 4]:
        train[column + '_' + str(number)] = train[column].str[number-1:number]
        test[column + '_' + str(number)] = test[column].str[number-1:number]
# %%
# add card number
train = hlp.add_card_id(train)
test = hlp.add_card_id(test)

# %%
# means
train['TransactionAmt_to_mean_card_id'] = train['TransactionAmt'] - \
    train.groupby(['Card_ID'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_std_card_id'] = train['TransactionAmt_to_mean_card_id'] / \
    train.groupby(['Card_ID'])['TransactionAmt'].transform('std')

train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / \
    train.groupby(['card1'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / \
    train.groupby(['card4'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / \
    train.groupby(['card1'])['TransactionAmt'].transform('std')
train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / \
    train.groupby(['card4'])['TransactionAmt'].transform('std')

test['TransactionAmt_to_mean_card_id'] = test['TransactionAmt'] - \
    test.groupby(['Card_ID'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_std_card_id'] = test['TransactionAmt_to_mean_card_id'] / \
    test.groupby(['Card_ID'])['TransactionAmt'].transform('std')

test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / \
    test.groupby(['card1'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / \
    test.groupby(['card4'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / \
    test.groupby(['card1'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / \
    test.groupby(['card4'])['TransactionAmt'].transform('std')


# %%
# new features
train['_weekday__hour'] = train['_Weekdays'] + train['_Hours']
cnt_day = train['_Days'].value_counts()
cnt_day = cnt_day / cnt_day.mean()
train['_count_rate'] = train['_Days'].map(cnt_day.to_dict())

train['_P_emaildomain__addr1'] = train['P_emaildomain'] + \
    '__' + train['addr1'].astype(str)

train['_card1__addr1'] = train['card1'].astype(
    str) + '__' + train['addr1'].astype(str)
train['_card2__addr1'] = train['card2'].astype(
    str) + '__' + train['addr1'].astype(str)
train['_cardID__addr1'] = train['Card_ID'] + '__' + train['addr1'].astype(str)


train['_amount_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt']
                             .astype(int)) * 1000).astype(int)
train['_amount_decimal_len'] = train['TransactionAmt'].apply(lambda x:
                                                             len(re.sub('0+$', '',
                                                                        str(x)).split('.')[1]))
train['_amount_fraction'] = train['TransactionAmt'].apply(lambda x:
                                                          float('0.' +
                                                                re.sub(r'^[0-9]|\.|0+$',
                                                                       '', str(x))))

test['_weekday__hour'] = test['_Weekdays'] + test['_Hours']
cnt_day = test['_Days'].value_counts()
cnt_day = cnt_day / cnt_day.mean()
test['_count_rate'] = test['_Days'].map(cnt_day.to_dict())

test['_P_emaildomain__addr1'] = test['P_emaildomain'] + \
    '__' + test['addr1'].astype(str)

test['_card1__addr1'] = test['card1'].astype(
    str) + '__' + test['addr1'].astype(str)
test['_card2__addr1'] = test['card2'].astype(
    str) + '__' + test['addr1'].astype(str)
test['_cardID__addr1'] = test['Card_ID'] + '__' + test['addr1'].astype(str)


test['_amount_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt']
                            .astype(int)) * 1000).astype(int)
test['_amount_decimal_len'] = test['TransactionAmt'].apply(lambda x:
                                                           len(re.sub('0+$', '',
                                                                      str(x)).split('.')[1]))
test['_amount_fraction'] = test['TransactionAmt'].apply(lambda x:
                                                        float('0.' +
                                                              re.sub(r'^[0-9]|\.|0+$', '', str(x))))


cols = ['ProductCD', 'card1', 'card2', 'card5', 'card6',
        'P_emaildomain', 'Card_ID']
# ,'card3','card4','addr1','dist2','R_emaildomain'

# %%
# amount mean&std
for f in cols:
    train[f'_amount_mean_{f}'] = train['TransactionAmt'] / train.groupby([f])['TransactionAmt']\
        .transform('mean')
    train[f'_amount_std_{f}'] = train['TransactionAmt'] / train.groupby([f])['TransactionAmt']\
        .transform('std')
    train[f'_amount_pct_{f}'] = (train['TransactionAmt'] - train[f'_amount_mean_{f}'])\
        / train[f'_amount_std_{f}']

    test[f'_amount_mean_{f}'] = test['TransactionAmt'] / test.groupby([f])['TransactionAmt']\
        .transform('mean')
    test[f'_amount_std_{f}'] = test['TransactionAmt'] / test.groupby([f])['TransactionAmt']\
        .transform('std')
    test[f'_amount_pct_{f}'] = (test['TransactionAmt'] - test[f'_amount_mean_{f}'])\
        / test[f'_amount_std_{f}']

# freq encoding
for f in cols:
    vc = train[f].value_counts(dropna=False)
    train[f'_count_{f}'] = train[f].map(vc)

    vc = test[f].value_counts(dropna=False)
    test[f'_count_{f}'] = test[f].map(vc)

# all V to 2 dimensions
V_cols = []
for col in train:
    if col.startswith('V'):
        V_cols.append(col)

sc = MinMaxScaler()

pca = PCA(n_components=2)  # 0.99
vcol_pca = pca.fit_transform(sc.fit_transform(train[V_cols].fillna(-1)))

train['_vcol_pca0'] = vcol_pca[:, 0]
train['_vcol_pca1'] = vcol_pca[:, 1]
train['_vcol_nulls'] = train[V_cols].isnull().sum(axis=1)

train.drop(V_cols, axis=1, inplace=True)

vcol_pca = pca.fit_transform(sc.fit_transform(test[V_cols].fillna(-1)))

test['_vcol_pca0'] = vcol_pca[:, 0]
test['_vcol_pca1'] = vcol_pca[:, 1]
test['_vcol_nulls'] = test[V_cols].isnull().sum(axis=1)

test.drop(V_cols, axis=1, inplace=True)
# %%
# drop columns with nans or correlation
keep_cols = [target, 'TransactionDT', 'TransactionID',
             'Card_ID', '_Hours', '_Days', '_Weekdays']

cols_to_drop_nulls = hlp.drop_columns_nan_null(train,
                                               keep_cols,
                                               drop_proportion=0.9)
cols_to_drop_corr = hlp.drop_columns_corr(train,
                                          keep_cols,
                                          drop_threshold=0.98)

total_drop_cols = list(set(cols_to_drop_nulls + cols_to_drop_corr))
print(f'{total_drop_cols} columns are going to be removed')
for col in total_drop_cols:
    if col in train and col in test:
        train.drop([col], axis=1, inplace=True)
        test.drop([col], axis=1, inplace=True)


# %%
print(train.shape)
print(test.shape)

# %%
# categorical columns into numbers
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
train_num = train.select_dtypes(include=numerics)
cat_columns = [col for col in train.columns if col not in train_num.columns]
if target in cat_columns:
    cat_columns.remove(target)

for col in cat_columns:
    if col in train.columns:
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) +
               list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))


# %%
# print info
print(train.info())
print(test.info())



# %%
# create X, Y for train and check feature importances


X = train.sort_values('TransactionDT').drop(
    [target, 'TransactionDT', 'TransactionID'], axis=1)
Y = train.sort_values('TransactionDT')[target].astype(float)
X_test = test.sort_values('TransactionDT').drop(
    ['TransactionDT', 'TransactionID'], axis=1)
# del train
# test = test[["TransactionDT", 'TransactionID']]
# %%
# Cleaning infinite values
X = hlp.clean_inf_nan(X)
X.fillna(-999, inplace=True)


# %%
# validation split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# %%
# params for model
params = {'num_leaves': 256,
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
          'colsample_bytree': 0.9,
          'n_estimators': 1000,
          'n_jobs': 7
          }
# %%
# do model
model = lgb.LGBMClassifier(**params)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          eval_metric='auc',
          verbose=100,
          early_stopping_rounds=100)

y_pred_valid = model.predict_proba(X_test)[:, 1]
# %%
print(hlp.fast_auc(y_pred_valid, y_test))

# %%
# plot feature importances
feature_imp = pd.DataFrame(sorted(
    zip(model.feature_importances_, X.columns)), columns=['Value', 'Feature'])

print(feature_imp.tail(50))
# plt.figure(figsize=(20, 10))
# sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
# plt.title('LightGBM Features (avg over folds)')
# plt.tight_layout()
# plt.show()

# %%
# not usefull features
# print(list(feature_imp.head(50)['Feature'].values))

# %%
# save train & test
train.to_csv(hlp.DATASETS_DEV_PATH + 'train9.csv', index=False)
test.to_csv(hlp.DATASETS_DEV_PATH + 'test9.csv', index=False)

# %%
# results summary
# 1 0.862 - raw dataset, transactionDT, transactionID not removed
# 2 0.86 - raw dataset, transactionDT, transactionID removed
# 3 0.859 - raw + nan count features
# 4 0.863 - raw + D_nulls
# 5 0.8627 - 4 + datetime
# 6 0.87 - 5 + card features
# 7 0.873 - 6 + cardID
# 8 0.878 - 7 + TransactionAmt to mean TST 09404
# 9 0.89 - 8 + https://www.kaggle.com/plasticgrammer/ieee-cis-fraud-detection-eda/output TST 09404
