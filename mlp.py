"""
Author: Qizhi Li
神经网络
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import matplotlib.pyplot as plt
import preprocess
import json
import pickle


class MLP(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.linear1 = nn.Linear(num_inputs, 128)
        self.linear_add1 = nn.Linear(128, 64)
        # self.linear_add2 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(64, 12)
        self.linear3 = nn.Linear(12, num_outputs)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.01)
        self.softmax = nn.Softmax()

    def forward(self, inputs):
        """
        前向传播
        :param inputs: tensor
                shape: (batch_size, 26)
        :return: tensor
                shape: (batch_size, 11)
        """
        out1 = self.sigmoid(self.linear1(inputs))
        out1 = self.dropout(out1)
        out_add1 = self.sigmoid(self.linear_add1(out1))
        out_add1 = self.dropout(out_add1)
        # out_add2 = self.sigmoid(self.linear_add2(out_add1))
        # out_add2 = self.dropout(out_add2)
        out2 = self.sigmoid(self.linear2(out_add1))
        out2 = self.dropout(out2)

        return self.softmax(self.linear3(out2))


def get_train_dev_test(inputs, labels):
    """
    获得训练集, 验证集, 测试集
    6: 2: 2
    :param inputs: tensor
            输入数据
    :param labels: tensor
            真实标签
    :return X_train: tensor
            训练集
    :return X_dev: tensor
            验证集
    :return X_test: tensor
            测试集
    :return y_train: tensor
            训练标签
    :return y_dev: tensor
            验证标签
    :return y_test: tensor
            测试标签
    """
    X_train, X_dt, y_train, y_dt = train_test_split(inputs, labels, test_size=0.4,
                                                    random_state=0)
    X_dev, X_test, y_dev, y_test = train_test_split(X_dt, y_dt, test_size=0.5,
                                                    random_state=0)
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def get_accuracy(y_hat, y, epsilon):
    """
    获得准确率
    判断y_hat每个元素与阈值的大小, 再与y做比较
    :param y_hat: tensor
            预测数据
    :param y: tensor
            真实数据
    :param epsilon: float
            阈值
    :return: float
            准确率
    """
    return ((y_hat >= epsilon).float() == y).float().mean().item()


def evaluate(model, data_iter, loss, epsilon):
    """
    评价模型
    :param model: Object
            模型
    :param data_iter: Object
            验证集或测试集的迭代对象
    :param loss: Object
            损失函数
    :param epsilon: float
            阈值,
            低于阈值的为0 (e.g. 不推荐今天穿这个衣服), 高于阈值的为1 (e.g. 推荐今天穿这个衣服)
    :return acc: float
            准确率
    :return: float
            平均损失
    """
    model.eval()
    loss_total = 0
    predict_all = torch.tensor([])
    labels_all = torch.tensor([])
    with torch.no_grad():
        for X, y in data_iter:
            y_hat = model(X)
            l = loss(y_hat, y)
            loss_total += l
            labels = y.data
            labels_all = torch.cat((labels_all, labels))
            predict_all = torch.cat((predict_all, y_hat))

    acc = get_accuracy(predict_all, labels_all, epsilon)

    return acc, loss_total / len(data_iter)


def train(inputs, labels, num_epochs, batch_size, lr, epsilon):
    """
    训练数据
    :param inputs: tensor
            shape: (num_days, 26)
    :param labels: tensor
            shape: (num_days, 11)
    :param num_epochs: int
            epoch数
    :param batch_size: int
            batch大小
    :param lr: float
            学习率
    :param epsilon: float
            阈值,
            低于阈值的为0 (e.g. 不推荐今天穿这个衣服), 高于阈值的为1 (e.g. 推荐今天穿这个衣服)
    """
    model = MLP(inputs.shape[1], labels.shape[1])
    X_train, X_dev, X_test, y_train, y_dev, y_test = get_train_dev_test(inputs, labels)

    train_dataset = Data.TensorDataset(X_train, y_train)
    dev_dataset = Data.TensorDataset(X_dev, y_dev)
    test_dataset = Data.TensorDataset(X_test, y_test)

    train_iter = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_iter = Data.DataLoader(dev_dataset, batch_size=batch_size)
    test_iter = Data.DataLoader(test_dataset, batch_size=batch_size)

    # loss = nn.MultiLabelSoftMarginLoss(reduction='mean')
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    i = 0
    train_accs, train_ls, dev_accs, dev_ls = [], [], [], []
    predict_all = torch.tensor([])
    labels_all = torch.tensor([])
    for epoch in range(num_epochs):
        model.train()
        for X, y in train_iter:
            y_hat = model(X)
            optimizer.zero_grad()
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            predict_all = torch.cat((predict_all, y_hat.detach()))
            labels_all = torch.cat((labels_all, y.data))

            if (i + 1) % 10 == 0:
                train_acc = get_accuracy(predict_all, labels_all, epsilon)
                dev_acc, dev_loss = evaluate(model, dev_iter, loss, epsilon)

                train_accs.append(train_acc)
                train_ls.append(l)
                dev_accs.append(dev_acc)
                dev_ls.append(dev_loss)

                predict_all = torch.tensor([])
                labels_all = torch.tensor([])

                print('iter %d, train accuracy %f, train loss %f, dev accuracy %f, dev loss %f' %
                      (i + 1, train_acc, l.item(), dev_acc, dev_loss))

            i += 1

    test_acc, test_loss = evaluate(model, test_iter, loss, epsilon)
    print('test accuracy %f, test loss %f' % (test_acc, test_loss))
    torch.save(model.state_dict(), './data/parameters_layer3.pkl')
    draw_fig(i // 10, train_accs, 'iter', 'train precision', 'data/train_acc_layer3.png')
    draw_fig(i // 10, train_ls, 'iter', 'train loss', 'data/train_ls_layer3.png')
    draw_fig(i // 10, dev_accs, 'iter', 'dev precision', 'data/dev_acc_layer3.png')
    draw_fig(i // 10, dev_ls, 'iter', 'dev loss', 'data/dev_ls_layer3.png')


def draw_fig(x, ys, x_label, y_label, file_path):
    """
    绘图
    :param x: int
            epochs
    :param ys: list
            纵轴
    :param x_label: str
            横轴标签
    :param y_label: str
            纵轴标签
    :param file_path: str
            保存路径
    """
    xs = range(x)
    plt.plot(xs, ys)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.show()
    plt.savefig(file_path)
    plt.close()


def get_dress(season, highest_temperature, lowest_temperature, morning_weather,
              night_weather, epsilon):
    """
    获得穿着
    :return:
    """
    model = MLP(26, 11)
    model.load_state_dict(torch.load('./data/parameters_layer2.pkl'))

    season_onehot_encoder = np.load('./data/season_onehot_encoder.npy')
    weather_onehot_encoder = np.load('./data/weather_onehot_encoder.npy')
    with open('./data/season2idx.json', 'r') as f:
        season2idx = json.load(f)
    with open('./data/weather2idx.json', 'r') as f:
        weather2idx = json.load(f)
    with open('./data/titles.pkl', 'rb') as f:
        titles = pickle.load(f)

    input_ = torch.FloatTensor([])
    try:
        season_onehot = torch.LongTensor(season_onehot_encoder[season2idx[season]])
        morning_weather_onehot = torch.LongTensor(weather_onehot_encoder[weather2idx[morning_weather]])
        night_weather_onehot = torch.LongTensor(weather_onehot_encoder[weather2idx[night_weather]])
        # scalar不转为long类型会报错
        highest_temperature = torch.tensor(highest_temperature, dtype=torch.long).reshape(-1)
        lowest_temperature = torch.tensor(lowest_temperature, dtype=torch.long).reshape(-1)
        input_ = torch.cat((input_, season_onehot, highest_temperature, lowest_temperature,
                            morning_weather_onehot, night_weather_onehot))
        # input_ = torch.LongTensor(input_)
    except KeyError:
        print('Please enter right data')
        exit()

    model.eval()
    with torch.no_grad():
        pred = model(input_)

    dress_idx = torch.nonzero((pred >= epsilon).float())  # 提取出非零的元素下标

    print('今日适合穿: ', end='')
    for idx in dress_idx:
        print(titles[idx], end=' ')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    if os.path.exists('./data/parameters_layer2.pkl'):
        parser = argparse.ArgumentParser(description="Enter data to get today's dress")
        parser.add_argument('-s', '--season', help='What season is today (春 夏 秋 冬)')
        parser.add_argument('-hi', '--highest_temperature',
                            help="What's the highest temperature today", type=float)
        parser.add_argument('-l', '--lowest_temperature',
                            help="What's the lowest temperature today", type=float)
        parser.add_argument('-m', '--morning_weather',
                            help="How is the weather this morning (中雨 多云 大暴雨 大雨 小雨 晴 暴雨 阴 阵雨 雷阵雨)",
                            type=str)
        parser.add_argument('-n', '--night_weather',
                            help="How is the weather this night (中雨 多云 大暴雨 大雨 小雨 晴 暴雨 阴 阵雨 雷阵雨)",
                            type=str)
        args = parser.parse_args()
        get_dress(args.season, args.highest_temperature, args.lowest_temperature,
                  args.morning_weather, args.night_weather, 0.1)
    else:
        print('Start training.')
        inputs, labels = preprocess.preprocessing()
        train(inputs, labels, 100, 16, 1e-2, 0.1)












