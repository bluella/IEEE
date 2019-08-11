#!/usr/bin/env python3
"""Module with all data manipulations
   P.S. Can be converted to .ipynb"""

# %%
# All the imports
import warnings
import multiprocessing
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
def my_csv_read(csv_file):
    """Solve function pickle issues
    https://stackoverflow.com/questions/8804830/
    python-multiprocessing-picklingerror-cant-pickle-type-function
    """
    return pd.read_csv(csv_file)

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
    test_transaction, sub = pool.map(my_csv_read, datasets)


# %%
# Merging transactions and identity
train = pd.merge(train_transaction, train_identity,
                 on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity,
                on='TransactionID', how='left')

# %%
# Merging test and train
target = 'isFraud'
test[target] = 'test'
df = pd.concat([train, test], axis=0, sort=False)
df = df.reset_index()

# %%
# print datasets info

# print(f"Train dataset has {df[df[target]!='test'].shape[0]} rows and \
# {df[df[target]!='test'].shape[1]} columns.")
# print(f"Test dataset has {df[df[target]=='test'].shape[0]} rows and \
# {df[df[target]=='test'].shape[1]} columns.")

# print(df.head())
# for column in df.columns:
#     print(column)


# %%
# delete heavy parts
del train_identity, train_transaction, test_identity, test_transaction, train, test


# %%
# add nans count as feature
V_columns = []
C_columns = []
M_columns = []
D_columns = []
id_columns = []
card_columns = []
other_columns = []

for column in df.columns:
    if 'card' in column:
        card_columns.append(column)
    elif 'id' in column:
        id_columns.append(column)
    elif 'D' in column:
        D_columns.append(column)
    elif 'M' in column:
        M_columns.append(column)
    elif 'C' in column:
        C_columns.append(column)
    elif 'V' in column:
        V_columns.append(column)
    else:
        other_columns.append(column)

df['card_nulls'] = df[card_columns].isnull().sum(axis=1)
df['id_nulls'] = df[id_columns].isnull().sum(axis=1)
df['D_nulls'] = df[D_columns].isnull().sum(axis=1)
df['M_nulls'] = df[M_columns].isnull().sum(axis=1)
df['C_nulls'] = df[C_columns].isnull().sum(axis=1)
df['V_nulls'] = df[V_columns].isnull().sum(axis=1)
df['other_nulls'] = df[other_columns].isnull().sum(axis=1)

# Lets add datetime features
df = hlp.add_datetime_info(df)

# card features
column_card_numbers = ['card1', 'card2', 'card3', 'card5']
for column in column_card_numbers:
    df[column] = df[column].map(str).fillna('NNNN').map(hlp.correct_card_id)
    for number in [1, 2, 3, 4]:
        df[column + '_' + str(number)] = df[column].str[number-1:number]

# add card number
df = hlp.add_card_id(df)

df['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]
df['device_version'] = df['DeviceInfo'].str.split('/', expand=True)[1]

df['OS_id_30'] = df['id_30'].str.split(' ', expand=True)[0]
df['version_id_30'] = df['id_30'].str.split(' ', expand=True)[1]

df['screen_width'] = df['id_33'].str.split('x', expand=True)[0]
df['screen_height'] = df['id_33'].str.split('x', expand=True)[1]

df['diff_adrr'] = df.addr1 - df.addr2
df['sum_adrr'] = df.addr1 + df.addr2

df['addr1'] = df['addr1'].fillna(0)
df['addr2'] = df['addr2'].fillna(0)

df['first_value_addr1'] = df['addr1'].astype(str).str[0:1].astype(float)
df['two_value_addr1'] = df['addr1'].astype(str).str[0:2].astype(float)

df['first_value_addr2'] = df['addr2'].astype(str).str[0:1].astype(float)
df['two_value_addr2'] = df['addr2'].astype(str).str[0:2].astype(float)

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
# just madness with renaming
# Device renaming
df.loc[df['device_name'].str.contains(
    'SM', na=False), 'device_name'] = 'Samsung'
df.loc[df['device_name'].str.contains(
    'SAMSUNG', na=False), 'device_name'] = 'Samsung'
df.loc[df['device_name'].str.contains(
    'GT-', na=False), 'device_name'] = 'Samsung'
df.loc[df['device_name'].str.contains(
    'Moto G', na=False), 'device_name'] = 'Motorola'
df.loc[df['device_name'].str.contains(
    'Moto', na=False), 'device_name'] = 'Motorola'
df.loc[df['device_name'].str.contains(
    'moto', na=False), 'device_name'] = 'Motorola'
df.loc[df['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
df.loc[df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
df.loc[df['device_name'].str.contains(
    'HUAWEI', na=False), 'device_name'] = 'Huawei'
df.loc[df['device_name'].str.contains(
    'ALE-', na=False), 'device_name'] = 'Huawei'
df.loc[df['device_name'].str.contains(
    '-L', na=False), 'device_name'] = 'Huawei'
df.loc[df['device_name'].str.contains(
    'Blade', na=False), 'device_name'] = 'ZTE'
df.loc[df['device_name'].str.contains(
    'BLADE', na=False), 'device_name'] = 'ZTE'
df.loc[df['device_name'].str.contains(
    'Linux', na=False), 'device_name'] = 'Linux'
df.loc[df['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
df.loc[df['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
df.loc[df['device_name'].str.contains(
    'ASUS', na=False), 'device_name'] = 'Asus'

df.loc[df.device_name.isin(df.device_name.value_counts()\
[df.device_name.value_counts() < 200].index), 'device_name'] = "Others"

df['browser_name'] = df['id_31']
df.loc[df['browser_name'].str.lower.contains(
    'sumsung', na=False), 'browser_name'] = 'sumsung'
df.loc[df['browser_name'].str.lower.contains(
    'safari', na=False), 'browser_name'] = 'safari'
df.loc[df['browser_name'].str.lower.contains(
    'opera', na=False), 'browser_name'] = 'opera'
df.loc[df['browser_name'].str.lower.contains(
    'google', na=False), 'browser_name'] = 'chrome'
df.loc[df['browser_name'].str.lower.contains(
    'firefox', na=False), 'browser_name'] = 'firefox'
df.loc[df['browser_name'].str.lower.contains(
    'edge', na=False), 'browser_name'] = 'edge'
df.loc[df['browser_name'].str.lower.contains(
    'android', na=False), 'browser_name'] = 'android'
df.loc[df['browser_name'].str.lower.contains(
    'chrome', na=False), 'browser_name'] = 'chrome'

df.loc[df.browser_name.isin(df.browser_name.value_counts()
                            [df.browser_name.value_counts() < 100].index), 'browser_name'] = "other"

df['browser_version'] = df['id_31'].str.map(hlp.get_floats_from_string).map(hlp.none_or_first)
# %%
# email madness
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum',
          'scranton.edu': 'other', 'netzero.net': 'other',
          'optonline.net': 'other', 'comcast.net': 'other',
          'cfl.rr.com': 'other', 'sc.rr.com': 'other',
          'suddenlink.net': 'other', 'windstream.net': 'other',
          'gmx.de': 'other', 'earthlink.net': 'other',
          'servicios-ta.com': 'other', 'bellsouth.net': 'other',
          'web.de': 'other', 'mail.com': 'other',
          'cableone.net': 'other', 'roadrunner.com': 'other',
          'protonmail.com': 'other', 'anonymous.com': 'other',
          'juno.com': 'other', 'ptd.net': 'other',
          'netzero.com': 'other', 'cox.net': 'other',
          'hotmail.co.uk': 'microsoft',
          'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
          'yahoo.es': 'yahoo', 'charter.net': 'spectrum',
          'live.com': 'microsoft', 'aim.com': 'aol',
          'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',
          'gmail.com': 'google', 'me.com': 'apple',
          'hotmail.com': 'microsoft',
          'hotmail.fr': 'microsoft',
          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo',
          'yahoo.de': 'yahoo',
          'live.fr': 'microsoft', 'verizon.net': 'yahoo',
          'msn.com': 'microsoft', 'q.com': 'centurylink',
          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo',
          'rocketmail.com': 'yahoo',
          'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo',
          'ymail.com': 'yahoo', 'outlook.com': 'microsoft',
          'embarqmail.com': 'centurylink',
          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo',
          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft',
          'aol.com': 'aol', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']

# https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest-579654
for c in ['P_emaildomain', 'R_emaildomain']:
    df[c + '_bin'] = df[c].map(emails)
    df[c + '_suffix'] = df[c].map(lambda x: str(x).split('.')[-1])
    df[c + '_suffix'] = df[c +
                           '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

# %%
# drop changed columns
df.drop(['id_31', 'id_33', 'id_30', 'DeviceInfo'], axis=1, inplace=True)
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
# check distos for specific set of columns
# d_cols = [c for c in train_transaction if c[0] == 'D']
# train[d_cols].head()
# sns.pairplot(train,
#              hue=target,
#             vars=d_cols)
# plt.show()

# %%
# drop columns
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
# print(cat_columns)
# cat_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17',
#             'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24',
#             'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
#             'id_30', 'id_31', 'id_32', 'id_33',
#             'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType',
#             'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4', 'P_emaildomain',
#             'R_emaildomain', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2',
#             'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9',
#             'P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3',
#             'R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3',
#             'Card_ID']

for col in cat_columns:
    if col in df.columns:
        le = LabelEncoder()
        le.fit(list(df[col].astype(str).values))
        df[col] = le.transform(list(df[col].astype(str).values))
        # test[col] = le.transform(list(test[col].astype(str).values))

# %%
# separate train and test
train, test = df[df[target] != 'test'], df[df[target]
                                           == 'test'].drop(target, axis=1)
del df

# %%
# train preparation


X = train.sort_values('TransactionDT').drop(
    [target, 'TransactionDT', 'TransactionID'], axis=1)
Y = train.sort_values('TransactionDT')[target].astype(float)
X_test = test.sort_values('TransactionDT').drop(
    ['TransactionDT', 'TransactionID'], axis=1)
# del train
test = test[["TransactionDT", 'TransactionID']]

# %%
# check
print(train.shape, test.shape)
print(X.shape, Y.shape, X_test.shape)
print(X.head(5))
print(Y.head(5))
# %%
# Cleaning infinite values to NaN
X = hlp.clean_inf_nan(X)
X_test = hlp.clean_inf_nan(X_test)

# %%
# Fill NaNs
X.fillna(-999, inplace=True)
X_test.fillna(-999, inplace=True)

# %%
# leaving only important features
# important_features = ['card1', 'Card_ID', 'TransactionAmt_to_std_card_id',
#                       'TransactionAmt_to_std_card1', 'TransactionAmt_to_mean_card_id',
#                       'TransactionAmt_to_mean_card1', 'TransactionAmt', 'card2', 'addr1',
#                       'dist1', '_Days', 'D15', 'D1', 'D4', 'D10', '_Hours', 'card1_4',
#                       'card1_3', 'card2_3', 'C13', 'card1_2', 'D11', 'V313', 'card2_2',
#                       'P_emaildomain', 'id_02', 'card5', 'V315', 'id_20', 'V307', 'V310',
#                       'C1', 'P_emaildomain_bin', 'card1_1', 'id_19', '_Weekdays', 'V314',
#                       'D5', 'D8', 'card5_3', 'D3', 'card2_1', 'V130', 'D_nulls', 'V312', 'C9',
#                       'card5_2', 'version_id_31', 'first_value_addr1', 'id_06']
# X = X[important_features].set_index('Card_ID')
# X_test = X_test[important_features].set_index('Card_ID')
X = X.set_index('Card_ID')
X_test = X_test.set_index('Card_ID')
# %%
# train params
n_fold = 5
# folds = TimeSeriesSplit(n_splits=n_fold)
folds = StratifiedKFold(n_splits=n_fold, shuffle=True)

# %%
# train lgb
# params = {'num_leaves': 256,
#           'min_child_samples': 79,
#           'objective': 'binary',
#           'max_depth': 13,
#           'learning_rate': 0.03,
#           "boosting_type": "gbdt",
#           "subsample_freq": 3,
#           "subsample": 0.9,
#           #   "bagging_seed": 11,
#           "metric": 'auc',
#           "verbosity": -1,
#           'reg_alpha': 0.3,
#           'reg_lambda': 0.3,
#           'colsample_bytree': 0.9,
#           'device_type': 'gpu'
#           # 'categorical_feature': cat_cols
#           }  # test_score = 0.9393 cvm = 0.93

params = {'num_leaves': 256,
          'min_child_samples': 79,
          'objective': 'binary',
          'max_depth': 13,
          'learning_rate': 0.03,
          "boosting_type": "gbdt",
          "subsample_freq": 1,
          "subsample": 0.1,
          "bagging_fraction": 0.1,
          "feature_fraction": 0.1,
          #   "bagging_seed": 11, # 'categorical_feature': cat_cols
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 1,
          'reg_lambda': 1,
          'colsample_bytree': 0.9,
          'device_type': 'gpu'
          }  # test_score = 0.9393 cvm = 0.93
# 'categorical_feature': cat_cols
# params = {'num_leaves': 491,
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
                                                 params=params, folds=folds, model_type='lgb',
                                                 eval_metric='auc', plot_feature_importance=True,
                                                 verbose=500, early_stopping_rounds=200,
                                                 n_estimators=6000, averaging='usual', n_jobs=7)


# %%
# saving results
test = test.sort_values('TransactionDT')
test['prediction_lgb'] = result_dict_lgb['prediction']
# in case of blendingo + result_dict_xgb['prediction']
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


# %%


# print(len(important_features))


# %%
# Bayesian optimization
# cut tr and val
# bayesian_tr_idx, bayesian_val_idx = train_test_split(
#     X, test_size=0.3, random_state=42, stratify=Y)
# bayesian_tr_idx = bayesian_tr_idx.index
# bayesian_val_idx = bayesian_val_idx.index

# %%
# lgb to optimize

# black box LGBM


# def LGB_bayesian(
#     # learning_rate,
#     num_leaves,
#     bagging_fraction,
#     feature_fraction,
#     min_child_weight,
#     min_data_in_leaf,
#     max_depth,
#     reg_alpha,
#     reg_lambda
# ):

#     # LightGBM expects next three parameters need to be integer.
#     num_leaves = int(num_leaves)
#     min_data_in_leaf = int(min_data_in_leaf)
#     max_depth = int(max_depth)

#     assert type(num_leaves) == int
#     assert type(min_data_in_leaf) == int
#     assert type(max_depth) == int

#     param = {
#         'num_leaves': num_leaves,
#         'min_data_in_leaf': min_data_in_leaf,
#         'min_child_weight': min_child_weight,
#         'bagging_fraction': bagging_fraction,
#         'feature_fraction': feature_fraction,
#         # 'learning_rate' : learning_rate,
#         'max_depth': max_depth,
#         'reg_alpha': reg_alpha,
#         'reg_lambda': reg_lambda,
#         'objective': 'binary',
#         'save_binary': True,
#         'seed': 1337,
#         'feature_fraction_seed': 1337,
#         'bagging_seed': 1337,
#         'drop_seed': 1337,
#         'data_random_seed': 1337,
#         'boosting_type': 'gbdt',
#         'verbose': 1,
#         'is_unbalance': False,
#         'boost_from_average': True,
#         'metric': 'auc',
#         'device_type': 'gpu',
#         'n_jobs': 7}

#     oof = np.zeros(len(X))
#     trn_data = lgb.Dataset(
#         X.iloc[bayesian_tr_idx].values, label=Y.iloc[bayesian_tr_idx].values)
#     val_data = lgb.Dataset(
#         X.iloc[bayesian_val_idx].values, label=Y.iloc[bayesian_val_idx].values)

#     clf = lgb.train(param, trn_data,  num_boost_round=50, valid_sets=[
#                     trn_data, val_data], verbose_eval=0, early_stopping_rounds=50)

#     oof[bayesian_val_idx] = clf.predict(
#         X.iloc[bayesian_val_idx].values, num_iteration=clf.best_iteration)

#     score = roc_auc_score(
#         X.iloc[bayesian_val_idx].values, oof[bayesian_val_idx])

#     return score


# bounds_LGB = {
#     'num_leaves': (31, 500),
#     'min_data_in_leaf': (20, 200),
#     'bagging_fraction': (0.1, 0.9),
#     'feature_fraction': (0.1, 0.9),
#     # 'learning_rate': (0.01, 0.3),
#     'min_child_weight': (0.00001, 0.01),
#     'reg_alpha': (1, 2),
#     'reg_lambda': (1, 2),
#     'max_depth': (5, 50),
# }

# init_points = 10
# n_iter = 15

# LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=42)
# print(LGB_BO.space.keys)

# print('-' * 130)

# with warnings.catch_warnings():
#     warnings.filterwarnings('ignore')
#     LGB_BO.maximize(init_points=init_points, n_iter=n_iter,
#                     acq='ucb', xi=0.0, alpha=1e-6)


# LGB_BO.max['target']
# LGB_BO.max['params']


# # %%
# print(Y.describe())


# %%
