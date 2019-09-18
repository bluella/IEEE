#!/usr/bin/env python3
"""Feature engineering
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


# %%
# Merging transactions and identity
train = pd.merge(train_transaction, train_identity,
                 on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity,
                on='TransactionID', how='left')

# %%
print(train.shape)
print(test.shape)
# %%
# Merging test and train
target = 'isFraud'
test[target] = 'test'
df = pd.concat([train, test], axis=0, sort=False)
df = df.reset_index()

# %%
# delete heavy parts
del train_identity, train_transaction, test_identity, test_transaction, train, test


# %%
# add nans count as feature

D_columns = []
for column in df.columns:
    if 'D' in column:
        D_columns.append(column)

df['D_nulls'] = df[D_columns].isnull().sum(axis=1)
# %%
# Lets add datetime features
df = hlp.add_datetime_info(df)

# %%
# card features
column_card_numbers = ['card1', 'card2', 'card3', 'card5']
for column in column_card_numbers:
    df[column] = df[column].map(str).fillna('NNNN').map(hlp.correct_card_id)
    for number in [1, 2, 3, 4]:
        df[column + '_' + str(number)] = df[column].str[number-1:number]
# %%
# add card number
df = hlp.add_card_id(df)

# %%
# means
df['TransactionAmt_to_mean_card_id'] = df['TransactionAmt'] - \
    df.groupby(['Card_ID'])['TransactionAmt'].transform('mean')

df['TransactionAmt_to_std_card_id'] = df['TransactionAmt_to_mean_card_id'] / \
    df.groupby(['Card_ID'])['TransactionAmt'].transform('std')

df['TransactionAmt_to_mean_card1'] = df['TransactionAmt'] / \
    df.groupby(['card1'])['TransactionAmt'].transform('mean')
df['TransactionAmt_to_mean_card4'] = df['TransactionAmt'] / \
    df.groupby(['card4'])['TransactionAmt'].transform('mean')
df['TransactionAmt_to_std_card1'] = df['TransactionAmt'] / \
    df.groupby(['card1'])['TransactionAmt'].transform('std')
df['TransactionAmt_to_std_card4'] = df['TransactionAmt'] / \
    df.groupby(['card4'])['TransactionAmt'].transform('std')


# %%
# drop columns with nans or correlation
keep_cols = [target, 'TransactionDT', 'TransactionID',
             'Card_ID', '_Hours', '_Days', '_Weekdays']
df = hlp.drop_columns_nan_null(df, df[df[target] != 'test'],
                               keep_cols,
                               drop_proportion=0.9)
df = hlp.drop_columns_corr(df, df[df[target] != 'test'],
                           keep_cols,
                           drop_threshold=0.98)

# %%
# categorical columns into numbers
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = df.select_dtypes(include=numerics)
cat_columns = [col for col in df.columns if col not in df_num.columns]
if target in cat_columns:
    cat_columns.remove(target)

for col in cat_columns:
    if col in df.columns:
        le = LabelEncoder()
        le.fit(list(df[col].astype(str).values))
        df[col] = le.transform(list(df[col].astype(str).values))
        # test[col] = le.transform(list(test[col].astype(str).values))

# %%
# remove total shit
remove_cols = ['id_10', 'id_04', 'V144', 'V8', 'id_12',
               'V238', 'V184', 'V224', 'V239', 'V262', 'V80', 'V64', 'V200',
               'V42', 'V51', 'V94', 'V34', 'V39', 'V6', 'id_34', 'V149', 'V147',
               'V228', 'V201', 'V2', 'V33', 'V60', 'V85', 'V243', 'V257', 'V260',
               'V43', 'V169', 'id_15', 'V81', 'V216', 'id_32', 'V189', 'V272',
               'V335', 'V11', 'V52',
               'V25', 'V26', 'V140', 'V210', 'V289', 'V217', 'V159', 'V168']
df.drop(remove_cols, axis=1, inplace=True)
# %%
# separate train and test
train, test = df[df[target] != 'test'], df[df[target]
                                           == 'test'].drop(target, axis=1)
del df

# %%
# print info
print(train.info())
print(test.info())
# %%
print(train.tail(5))
# %%
# save train & test
train.to_csv(hlp.DATASETS_DEV_PATH + 'train8.csv', index=False)
test.to_csv(hlp.DATASETS_DEV_PATH + 'test8.csv', index=False)
# %%
# create X, Y for train and check feature importances

# Under question
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
# results summary
# 1 0.862 - raw dataset, transactionDT, transactionID not removed
# 2 0.86 - raw dataset, transactionDT, transactionID removed
# 3 0.859 - raw + nan count features
# 4 0.863 - raw + D_nulls
# 5 0.8627 - 4 + datetime
# 6 0.87 - 5 + card features
# 7 0.873 - 6 + cardID
# 8 0.878 - 7 + TransactionAmt to mean
