# coding:utf-8

# -----------------------------------------------------
# 读取数据，并把数据划分为train set和test set
# 输入：三元组形式，SPO，并以逗号区分
# 输出：trian和test
# ------------------------------------------------------

import numpy as np
import pandas as pd
from collections import Counter
import operator
import random
# from sklearn.naive_bayes import GaussianNB

# 把数据读取成：[用户id，物品id，评分]
def readData(filepath):
    if filepath == None:
        print('请输入地址！')
        return None
    ori_data = []
    # 读取的数据，每个三元组有且仅出现一次
    with open(filepath, 'r', encoding='UTF-8') as f:
        while f.readline():
            ori_data.append(f.readline().strip().split(','))
    # print(ori_data[:5])
    data = []
    items = set()
    for line in ori_data:
        try:
            if line[1] == 'Participate_In':
                data.append([line[0], line[2], 1])
                items.add(line[2])
        except:
            i = 1
    users = []
    for i in data:
        users.append(i[0])
    users_num = Counter(users)
    users_num = sorted(users_num.items(), key=operator.itemgetter(1), reverse=True)

    data_df = pd.DataFrame(data, columns=['vol', 'opp', 'score'])
    users_num_df = pd.DataFrame(users_num, columns=['vol', 'num'])
    data_df = pd.merge(data_df, users_num_df, how='left', on='vol')
    data_df = data_df.sort_values(by='num', ascending=False).reset_index(drop=True)
    output = data_df.values
    return output, list(items)

# 考虑到每个用户的数据量不同，对每个用户进行随机M划分，即划分成M份
# 一份去测试集，M-1份去训练集
# 自始至终没有求item的集合，只有每个用户买的物品
def SplitData(data, M=5, k=0, seed=0):
    # 不能指定seed
    # random.seed(seed)
    # 按志愿者和次数排序，逆序
    data_df = pd.DataFrame(data, columns=['vol', 'opp', 'score', 'num'])
    data_df = data_df.sort_values(by=['num', 'vol'], ascending=False).reset_index(drop=True)
    # users去重
    data2 = data_df.values

    # 用户数目
    users_num = len(set(data2[:, 0]))

    i = 0
    test = []
    train = []
    while i < users_num:

        num = int(data2[0][3])
        test_num = int(round(num / float(M)))

        # 对用户随机选取test_num个元素进入test
        for ii in range(test_num):
            index = random.randint(0, num - 1 - ii)
            test.append(data2[index][:3])
            # np.delete使用一定要小心，3个参数不能少，少最后一个参数，默认合一起
            data2 = np.delete(data2, index, 0)

        # 用户剩余数据进入train
        for ii in range(num - test_num):
            train.append(data2[0][:3])
            data2 = np.delete(data2, 0, 0)

        i += 1
    return train, test    # [[user,item,rating],]