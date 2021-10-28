"""
Author: Qizhi Li
对数据预处理, 包括:
    1. 将日期分为春夏秋冬, 简洁而言,
       11 - 02 为冬天, 02 - 05 为春天, 05 - 08为夏天, 08 - 11 为秋天
    2. 将春夏秋冬变为 one-hot 编码
    3. 将天气变为 one-hot 编码
"""
import os
import torch
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import json
import pickle


def load_data(file_path):
    """
    加载数据
    :param file_path: str
            文件路径
    :return dates: list
            时间
    :return highest_temps: ndarray
            shape: (num_days, 1)
            每天的最高气温
    :return lowest_temps: ndarray
            shape: (num_days, 1)
            每天的最低气温
    :return weather_mornings: list
            每天早上的天气
    :return weather_nights: list
            每天晚上的天气
    :return labels: ndarray
            shape: (num_days, 11)
            标签
    :return titles: list
            表头
    """
    data = pd.read_csv(file_path, encoding='gb2312')

    dates = np.array(data['日期']).tolist()  # 将日期转为列表, 方便后续对日期进行操作
    highest_temps = np.array(data['最高气温']).reshape(-1, 1)
    lowest_temps = np.array(data['最低气温']).reshape(-1, 1)
    weather_mornings = np.array(data['早晨天气']).tolist()
    weather_nights = np.array(data['晚间天气']).tolist()
    labels = np.array(data.iloc[:, 5:])
    titles = data.columns[5:].tolist()

    return dates, highest_temps, lowest_temps, weather_mornings, weather_nights, labels, titles


def get_season(dates):
    """
    将日期分为春夏秋冬
    :param dates: list
            时间
    :return seasons: list
            季节, 11 - 02 为冬天, 02 - 05 为春天, 05 - 08为夏天, 08 - 11 为秋天
    """
    reg = '[0-9]+月'
    seasons = []
    months = []
    for date in dates:
        months.append(int(re.search(reg, date).group().rstrip('月')))

    for month in months:
        if 2 <= month < 5:
            seasons.append('春')
        elif 5 <= month < 8:
            seasons.append('夏')
        elif 8 <= month < 11:
            seasons.append('秋')
        else:
            seasons.append('冬')

    return seasons


def get_id_char_mapping(char_list):
    """
    获得id与词的映射关系
    :param char_list: list
            词列表
    :return idx2char: dict
            {id1: 'char1', id2: 'char2', ...}
            id与词之间的映射关系
    :return char2idx: dict
            {'char1': id1, 'char2': id2, ...}
            词与id之间的映射关系
    """
    idx2char, char2idx = {}, {}
    char_set = set(char_list)  # 去重
    for i, char_ in enumerate(char_set):
        idx2char[i] = char_
        char2idx[char_] = i

    return idx2char, char2idx


def get_seq2idx(sequence, char2idx):
    """
    将序列数据映射为id
    :param sequence: list
            序列数据
    :param char2idx: dict
            {'char1': id1, 'char2': id2, ...}
            词与id之间的映射关系
    :return sequence2idx: list
            映射为id后的序列数据
    """
    sequence2idx = []
    for char_ in sequence:
        sequence2idx.append(char2idx[char_])

    return sequence2idx


def onehot_encode_seq(onehot_encoder, sequence):
    """
    对序列进行one-hot编码
    :param onehot_encoder: ndarray
            onehot编码器
    :param sequence: list
            需要编码的序列
    :return onehot: ndarray
            onehot编码后的序列
    """
    onehot = np.zeros((len(sequence), len(onehot_encoder)))

    for i, id_ in enumerate(sequence):
        onehot[i] = onehot_encoder[id_]

    return onehot


def encode_data(seasons, weather_mornings, weather_nights):
    """
    对数据进行编码, 将季节和天气编码为one-hot
    季节: shape: (4, 4)
    天气: shape: ()
    :param seasons: list
            季节
    :param weather_mornings: list
            早晨天气
    :param weather_nights: list
            晚间天气
    :return season_onehot: ndarray
            shape: (num_days, 4)
            季节的one-hot编码
    :return weather_mornings_onehot: ndarray
            shape: (num_days, 10)
            早晨天气的one-hot编码
    :return weather_nights_onehot: ndarray
            shape: (num_days, 10)
            晚间天气的one-hot编码
    """
    onehot_encoder = OneHotEncoder()  # one-hot编码器

    idx2season, season2idx = get_id_char_mapping(seasons)
    season_onehot_encoder = onehot_encoder.fit_transform(
        np.array(list(idx2season.keys())).reshape(-1, 1)
    ).toarray()  # 获得season的one-hot编码
    season_seq2idx = get_seq2idx(seasons, season2idx)  # 将sequence转为id

    # 根据id与one-hot的映射关系将sequence转为one-hot编码
    season_onehot = onehot_encode_seq(season_onehot_encoder, season_seq2idx)

    weather = []
    weather.extend(weather_mornings)
    weather.extend(weather_nights)
    idx2weather, weather2idx = get_id_char_mapping(weather)
    weather_onehot_encoder = onehot_encoder.fit_transform(
        np.array(list(idx2weather.keys())).reshape(-1, 1)
    ).toarray()

    weather_mornings_seq2idx = get_seq2idx(weather_mornings, weather2idx)
    weather_nights_seq2idx = get_seq2idx(weather_nights, weather2idx)
    weather_mornings_onehot = onehot_encode_seq(weather_onehot_encoder,
                                                weather_mornings_seq2idx)
    weather_nights_onehot = onehot_encode_seq(weather_onehot_encoder,
                                              weather_nights_seq2idx)

    if not os.path.exists('./data/season_onehot_encoder.npy'):
        np.save('./data/season_onehot_encoder.npy', season_onehot_encoder)

    if not os.path.exists('./data/weather_onehot_encoder.npy'):
        np.save('./data/weather_onehot_encoder.npy', weather_onehot_encoder)
        
    if not os.path.exists('./data/season2idx.json'):
        with open('./data/season2idx.json', 'w') as f:
            json.dump(season2idx, f)

    if not os.path.exists('./data/weather2idx.json'):
        with open('./data/weather2idx.json', 'w') as f:
            json.dump(weather2idx, f)

    return season_onehot, weather_mornings_onehot, weather_nights_onehot


def preprocessing():
    """
    数据预处理, 包括
        1. 数据编码
        2. 数据拼接
    :return inputs: tensor
            输入层数据
    :return labels: tensor
            真实标签
    """
    file_path = './data/data_1.csv'
    dates, highest_temps, lowest_temps, weather_mornings, weather_nights, labels, titles = load_data(file_path)
    seasons = get_season(dates)
    season_onehot, weather_mornings_onehot, weather_nights_onehot = encode_data(seasons,
                                                                                weather_mornings,
                                                                                weather_nights)

    inputs = np.hstack((season_onehot, highest_temps))
    inputs = np.hstack((inputs, lowest_temps))
    inputs = np.hstack((inputs, weather_mornings_onehot))
    inputs = np.hstack((inputs, weather_nights_onehot))

    if not os.path.exists('./data/titles.pkl'):
        with open('./data/titles.pkl', 'wb') as f:
            pickle.dump(titles, f)

    return torch.FloatTensor(inputs), torch.FloatTensor(labels)
