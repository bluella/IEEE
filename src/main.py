#!/usr/bin/env python3
"""Module with all data manipulations
   P.S. Can be converted to .ipynb"""

# %%
# All the imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, KFold
from src.lib import helpers as hlp


# %%
# Load datasets
# print(f'{hlp.DATASETS_ORIGINAL_PATH}train_identity.csv')
train_identity = pd.read_csv(f'{hlp.DATASETS_ORIGINAL_PATH}train_identity.csv')
train_transaction = pd.read_csv(
    f'{hlp.DATASETS_ORIGINAL_PATH}train_transaction.csv')
test_identity = pd.read_csv(f'{hlp.DATASETS_ORIGINAL_PATH}test_identity.csv')
test_transaction = pd.read_csv(
    f'{hlp.DATASETS_ORIGINAL_PATH}test_transaction.csv')
sub = pd.read_csv(f'{hlp.DATASETS_PRED_PATH}sample_submission.csv')
# let's combine the data and work with the whole dataset

# %%
# Merging transactions and identity
train = pd.merge(train_transaction, train_identity,
                 on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity,
                on='TransactionID', how='left')


# %%
# print datasets info
print(f'Train dataset has {train.shape[0]} rows and {train.shape[1]} columns.')
print(f'Test dataset has {test.shape[0]} rows and {test.shape[1]} columns.')

print(train.head())
print(test.head())
# print(hlp.resumetable(train))

# %%
# delete heavy parts
del train_identity, train_transaction, test_identity, test_transaction

# %%
# Lets add datetime features
train = hlp.add_datetime_info(train)
test = hlp.add_datetime_info(test)
# train['_Days'].describe()
# sns.distplot(train['_Days'].dropna())
# plt.title('Distribution of _Hours variable')
# %%
# check specific feature# %%

# train['id_11'].value_counts(dropna=False, normalize=True).head()
# sns.distplot(train['id_07'].dropna())
# plt.title('Distribution of id_07 variable')

# %%
# compare train and test distributions

# sns.distplot(train['TransactionDT'], label='train')
# sns.distplot(test['TransactionDT'], label='test')
# plt.legend()
# plt.title('Distribution of transactiond dates')


# %%
# drop columns

one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]
many_null_cols = [col for col in train.columns if
                  train[col].isnull().sum() / train.shape[0] > 0.9]
many_null_cols_test = [col for col in test.columns if
                       test[col].isnull().sum() / test.shape[0] > 0.9]
big_top_value_cols = [col for col in train.columns if
                      train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
big_top_value_cols_test = [col for col in test.columns
                           if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
cols_to_drop = list(set(many_null_cols +
                        many_null_cols_test +
                        big_top_value_cols +
                        big_top_value_cols_test +
                        one_value_cols +
                        one_value_cols_test))
cols_to_drop.remove('isFraud')
print(len(cols_to_drop), ' columns were removed')
train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)

# %%
# categorical columns into numbers
cat_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17',
            'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24',
            'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
            'id_30', 'id_31', 'id_32', 'id_33',
            'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType',
            'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4', 'P_emaildomain',
            'R_emaildomain', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2',
            'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',
            'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3',
            'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']

for col in cat_cols:
    if col in train.columns:
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) +
               list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))


# %%
# train preparation


X = train.sort_values('TransactionDT').drop(
    ['isFraud', 'TransactionDT', 'TransactionID'], axis=1)
Y = train.sort_values('TransactionDT')['isFraud']
X_test = test.sort_values('TransactionDT').drop(
    ['TransactionDT', 'TransactionID'], axis=1)
del train
test = test[["TransactionDT", 'TransactionID']]

# %%
# Cleaning infinite values to NaN
X = hlp.clean_inf_nan(X)
X_test = hlp.clean_inf_nan(X_test)

# %%
# train params
n_fold = 5
folds = TimeSeriesSplit(n_splits=n_fold)
folds = KFold(n_splits=n_fold)

# %%
# lgb
params = {'num_leaves': 256,
          'min_child_samples': 79,
          'objective': 'binary',
          'max_depth': 13,
          'learning_rate': 0.03,
          "boosting_type": "gbdt",
          "subsample_freq": 3,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3,
          'reg_lambda': 0.3,
          'colsample_bytree': 0.9,
          #'categorical_feature': cat_cols
         }

result_dict_lgb = hlp.train_model_classification(X=X, X_test=X_test, y=Y,
                                                 params=params, folds=folds, model_type='lgb',
                                                 eval_metric='auc', plot_feature_importance=False,
                                                 verbose=500, early_stopping_rounds=200,
                                                 n_estimators=6000, averaging='usual', n_jobs=-1)

# %%
# xgb
# xgb_params = {'eta': 0.04,
#               'max_depth': 5,
#               'subsample': 0.85,
#               'objective': 'binary:logistic',
#               'eval_metric': 'auc',
#               'silent': True,
#               'nthread': -1,
#               'tree_method': 'gpu_hist'}
# result_dict_xgb = hlp.train_model_classification(X=X, X_test=X_test, y=Y,
#                                                  params=xgb_params, folds=folds, model_type='xgb',
#                                                  eval_metric='auc', plot_feature_importance=False,
#                                                  verbose=500, early_stopping_rounds=200,
#                                                  n_estimators=5000, averaging='usual')
# %%
# saving results

test = test.sort_values('TransactionDT')
test['prediction_lgb'] = result_dict_lgb['prediction']
# in case of blendingo + result_dict_xgb['prediction']
sub['isFraud'] = pd.merge(sub, test, on='TransactionID')['prediction_lgb']
sub.to_csv(f'{hlp.DATASETS_PRED_PATH}submission.csv', index=False)




#%%
# blending
# blend_base = pd.read_csv(f'{hlp.DATASETS_PRED_PATH}submission_TST_09393_CVM_09297.csv')
# print(blend_base.head())
# print(blend_base.describe())

# sub['isFraud'] = (sub['isFraud'] + blend_base['isFraud'])/2
# sub.to_csv(f'{hlp.DATASETS_PRED_PATH}submission.csv', index=False)


#%%
