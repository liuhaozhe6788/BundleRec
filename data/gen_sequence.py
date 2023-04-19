import gc
import os.path
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm


def read_processed_bundle_data(path: str,
                               to_path: str,
                               user: str,
                               time: str,
                               spare_features: list,
                               dense_features: list,
                               usecols: list = None,
                               check_cols: list = None,
                               sep=','):
    """
    :param dense_features: 连续特征
    :param spare_features: 类别特征
    :param path: data_path
    :param user: user_col_name
    :param time: time_col_name
    :param usecols: Use None to read all columns.
    :param check_cols: delete null rows
    :param sep:
    :return: Returns a DataFrameGroupy by uid.
    """
    if check_cols is None:
        check_cols = []
    df = pd.read_csv(path, sep=sep, usecols=usecols)
    print("loaded data of {} rows".format(df.shape[0]))
    for col in usecols:
        null_num = df[col].isnull().sum()
        if null_num > 0:
            print("There are {} nulls in {}.".format(null_num, col))
            df = df[~df[col].isnull()]
    print("buy:{}, unbuy:{}".format(df[df['impression_result'] == 1].shape[0], df[df['impression_result'] == 0].shape[0]))
    df['register_time'] = pd.to_datetime(df["register_time"]).apply(lambda x: int(x.timestamp()))
    df = reduce_mem(df)

    lbes = {feature: LabelEncoder() for feature in spare_features}
    for feature in spare_features:
        df[feature] = lbes[feature].fit_transform(df[feature])
    with open(to_path+"lbes.pkl", 'wb') as file:
        pickle.dump(lbes, file)
    mms = MinMaxScaler()
    df[dense_features] = mms.fit_transform(df[dense_features])
    with open(to_path+"mms.pkl", 'wb') as file:
        pickle.dump(mms, file)

    grouped = df.sort_values(by=[time]).groupby(user)
    return grouped


def transform2sequences(grouped, default_col, sequence_col):
    """
    :param grouped:
    :param default_col:
    :param sequence_col: columns needed to generate sequences.
    :return: DataFrame
    """
    # TODO: to be updated
    df = pd.DataFrame(
        data={
            "uid": list(grouped.groups.keys()),
            **{col_name: grouped[col_name].apply(lambda x: x.iloc[0]) for col_name in default_col[1:]},
            **{col_name: grouped[col_name].apply(list) for col_name in sequence_col},
        }
    )
    return df


def create_fixed_sequences(values: list, sequence_length):
    sequences = []
    for end_index in range(len(values)):
        valid_len = min(sequence_length, end_index + 1)
        seq = values[end_index + 1 - valid_len:end_index + 1]
        sequences.append(seq.copy())
    return sequences


def split_train_val_test_by_user(df: pd.DataFrame, user: str, time: str):
    """
    Use last user record as test data.
    :param df:
    :param user: user_col_name
    :param time: time_col_name
    :return: DataFrame
    """
    grouped = df.sort_values(by=time).groupby(user)
    # 过滤用户行为数量小于等于3的用户
    grouped = grouped.filter(lambda x: x.shape[0] > 2)
    train = grouped.apply(lambda x: x[: -2])
    val = grouped.apply(lambda x: x[-2: -1])
    test = grouped.apply(lambda x: x[-1:])
    return train, val, test


def split_train_val_test_by_time(df: pd.DataFrame, user: str, time: str):
    """
    Use last user record as test data.
    :param df:.
    :param user: user_col_name
    :param time: time_col_name
    :return: DataFrame
    """
    df = df.sort_values(by=time)
    df.drop(time, axis=1, inplace=True)
    rows = df.shape[0]
    th = int(rows * 0.8)
    th2 = int(rows * 0.9)
    train = df[:th]
    val = df[th:th2]
    test = df[th2:]
    return train, val, test


def gen_bundle_data():
    path = "./bundle_time20/"
    if not os.path.exists(path):
        os.mkdir(path)
    user_default_col = ['uid', 'register_country', 'register_time', 'is_visitor', 'register_device',
                        'register_device_brand', 'register_os', 'gender']
    user_changeable_col = ['ftime', 'bundle_id', 'bundle_price', 'impression_result', 'island_no', 'spin_stock',
                           'coin_stock', 'diamond_stock', 'island_complete_gap_coin',
                           'island_complete_gap_building_cnt', 'tournament_rank', 'login_cnt_1d', 'ads_watch_cnt_1d',
                           'register_country_arpu', 'life_time', 'star', 'battle_pass_exp', 'is_up_to_date_version',
                           'pet_heart_stock', 'pet_exp_stock', 'friend_cnt', 'social_friend_cnt', 'pay_amt', 'pay_cnt',
                           'pay_per_day', 'pay_mean']
    spare_features = ['uid', 'is_visitor', 'gender', 'is_up_to_date_version', 'register_country', 'register_device',
                      'register_device_brand', 'register_os', 'bundle_id']
    dense_features = ['register_time', 'bundle_price', 'island_no', 'spin_stock', 'coin_stock',
                      'diamond_stock', 'island_complete_gap_coin', 'island_complete_gap_building_cnt',
                      'tournament_rank', 'login_cnt_1d', 'ads_watch_cnt_1d', 'register_country_arpu',
                      'life_time', 'star', 'battle_pass_exp', 'pet_heart_stock', 'pet_exp_stock', 'friend_cnt',
                      'social_friend_cnt', 'pay_amt', 'pay_cnt', 'pay_per_day', 'pay_mean']

    grouped = read_processed_bundle_data(path="./raw_data/bundle/brs_daily_20211101_20211230.csv",
                                         user='uid',
                                         time='ftime',
                                         spare_features=spare_features,
                                         dense_features=dense_features,
                                         usecols=user_default_col + user_changeable_col,
                                         check_cols=['bundle_id'],
                                         to_path=path)

    df = transform2sequences(grouped, user_default_col, user_changeable_col)

    sequence_length = 8
    for col_name in user_changeable_col:
        df[col_name] = df[col_name].apply(lambda x: create_fixed_sequences(x, sequence_length=sequence_length))

    df = df.explode(column=user_changeable_col, ignore_index=True)
    df['cur_time'] = df['ftime'].apply(lambda x: x[-1])
    train, val, test = split_train_val_test_by_time(df, 'uid', 'cur_time')
    train.to_csv(path+"train_data.csv", index=False, sep=',')
    val.to_csv(path+"val_data.csv", index=False, sep=',')
    test.to_csv(path+"test_data.csv", index=False, sep=',')
    print("Done")


def reduce_mem(df):
    starttime = time.time()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in tqdm(df.columns):
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if pd.isnull(c_min) or pd.isnull(c_max):
                continue
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
    print('-- Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction),time spend:{:2.2f} min'.format(end_mem, 100 * (start_mem - end_mem) / start_mem, (time.time() - starttime) / 60))
    gc.collect()
    return df


def plot_user_click_count_his(path: str, user: str):
    df = pd.read_csv(path)
    grouped = df.groupby(user)
    shapes = grouped.apply(lambda x: x.shape[0])
    from matplotlib import pyplot as plt
    plt.hist(shapes.values, bins='auto', edgecolor="r", histtype="step")
    plt.show()


if __name__ == '__main__':
    gen_bundle_data()

