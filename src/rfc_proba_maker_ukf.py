import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold



def add_category(train, test):
    train["category"] = 0
    test["category"] = 0
    
    # train segments with more then 9 open channels classes
    train.loc[2_000_000:2_500_000-1, 'category'] = 1
    train.loc[4_500_000:5_000_000-1, 'category'] = 1
    
    # test segments with more then 9 open channels classes (potentially)
    test.loc[500_000:600_000-1, "category"] = 1
    test.loc[700_000:800_000-1, "category"] = 1
    
    return train, test


def read_input():
    train = pd.read_csv("../data/unscentedkalmanfilteredsignals-for-ionchannel/UnscentedKalmanTrain.csv")
    test = pd.read_csv("../data/unscentedkalmanfilteredsignals-for-ionchannel/UnscentedKalmanTest.csv")
    return train, test

def shifted_feature_maker(df, periods=[1], add_minus=True):
    periods = np.asarray(periods, dtype=np.int32)
        
    if add_minus:
        periods = np.append(periods, -periods)

    df_transformed = df.copy()

    for p in periods:
        df_transformed[f"shifted_{p}"] = df_transformed["signal"].shift(periods=p, fill_value=True)

    return df_transformed

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        if col!='open_channels':
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def save_submission(y_test):
    submission = pd.read_csv("../input/liverpool-ion-switching/sample_submission.csv")
    submission["open_channels"] = np.asarray(y_test, dtype=np.int32)
    submission.to_csv("submission.csv", index=False, float_format="%.4f")

GROUP_SIZE = 4000
SPLITS = 6
nn_epochs = 180
nn_batch_size = 16

def batching(df, group_size):
    df['group'] = df.groupby(df.index//group_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df

train, test = read_input()
print("データを読み込みました.")
open_channels = train.open_channels
train, test = add_category(train, test)

train = shifted_feature_maker(train, periods=range(1, 15))
test = shifted_feature_maker(test, periods=range(1, 15))
print("特徴量を作りました.")

train = batching(train, GROUP_SIZE)
test = batching(test, GROUP_SIZE)
grouping = train.group
unique_group = grouping.unique()

FOLD = 5
kf = KFold(n_splits=FOLD, shuffle=True, random_state=71)

train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

train = train.drop(["time", "open_channels"], axis=1)
test = test.drop(["time"], axis=1).values

rfc_proba_train = np.zeros((len(train), 11))
rfc_proba_test = np.zeros((len(test), 11))
submit_data = np.zeros(len(test))

for n_fold, (tr_group_idx, val_group_idx) in enumerate(kf.split(unique_group)):
    tr_groups, va_groups = unique_group[tr_group_idx], unique_group[val_group_idx]
    is_tr = grouping.isin(tr_groups)
    is_va = grouping.isin(va_groups)
    tr_x, va_x = train[is_tr], train[is_va]
    tr_y, va_y = open_channels[is_tr], open_channels[is_va]
    print(f'Our training dataset shape is {tr_x.shape}')
    print(f'Our validation dataset shape is {va_x.shape}')

    clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=17,
            max_features=10,
            random_state=71,
            n_jobs=10,
            verbose=2)
    clf.fit(tr_x, tr_y)
    preds_f = clf.predict_proba(va_x)
    rfc_proba_train[is_va] += preds_f
    preds_f = clf.predict_proba(test)
    rfc_proba_test += preds_f
    submit_data += clf.predict(test)
    
submit_data /= FOLD
rfc_proba_test /= FOLD
save_submission(submit_data)
np.save('rfc_proba_train.npy', rfc_proba_train)
np.save('rfc_proba_test.npy', rfc_proba_test)
