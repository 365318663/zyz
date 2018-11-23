# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 20:36:48 2018

@author: lanlandetian
"""

import UserCF
import UserCF_IIF
import ItemCF
import ItemCF_IUF
import random
import Evaluation
import LFM
# import transE

import imp
imp.reload(UserCF)
imp.reload(ItemCF)
imp.reload(ItemCF_IUF)
imp.reload(Evaluation)
imp.reload(LFM)


def readData():
    data = []
    fileName = './data_new.txt'
    fr = open(fileName,'r')
    for line in fr.readlines():
        lineArr = line.strip().split()
        data.append([lineArr[0], lineArr[1], 1.0])
    return data

def items():
    with open('G:/WSPN/Knowledge Graph/beitaiping/SPO/data.txt', 'r') as f:
        items = set()
        for line in f.readlines():
            items.add(line.split(" ")[1].strip())
        return items
    
def SplitData(data):
    ret = transform(data)
    train = dict()
    test = dict()

    for user in ret:
        if(len(ret[user])==1):
            train[user]=ret[user]
        elif(len(ret[user])==2):
            i = 0
            test[user] = dict()
            train[user] = dict()
            for item in ret[user]:
                if(i>0):
                    test[user][item] = ret[user][item]
                else:
                    train[user][item] = ret[user][item]
                i+=1
        elif(len(ret[user]) == 3):
            i = 0
            test[user] = dict()
            train[user] = dict()
            for item in ret[user]:
                if (i >1 ):
                    test[user][item] = ret[user][item]
                else:
                    train[user][item] = ret[user][item]
                i+=1
        else:
            i = 0
            test[user] = dict()
            train[user] = dict()
            for item in ret[user]:
                if (i >0.7* len(ret[user])):
                    test[user][item] = ret[user][item]
                else:
                    train[user][item] = ret[user][item]
                i+=1
    return ret,train,test

    
# 将列表形式数据转换为dict形式
def transform(oriData):
    ret = dict()
    for user,item,rating in oriData:
        if user not in ret:
            ret[user] = dict()
        ret[user][item] = rating
    return ret
    
if __name__ == '__main__':
    data = readData()
    precision = 0
    recall = 0
    coverage = 0
    popularity = 0
    _, train, test = SplitData(data)
    print("train_num:"+str(len(train))+" test_num:"+str(len(test)))
    W = transE.ItemSimilarity(items())
    N = 1
    result = transE.Recommendation(test.keys(), train, W,N)
    precision = Evaluation.Precision(train, test, result, N)
    recall = Evaluation.Recall(train, test, result, N)
    coverage = Evaluation.Coverage(train, test, result, N)
        # popularity += Evaluation.Popularity(train, test, result, N)

    # popularity /= iteration

    # 输出结果
    print('TransE precision = %f' % precision)
    print('TransE recall = %f' % recall)
    print('Transe coverage = %f' % coverage)
##########################################################################
    W = ItemCF_IUF.ItemSimilarity(train)
    N = 1
    result = ItemCF_IUF.Recommendation(test.keys(), train, W,N)
    precision = Evaluation.Precision(train, test, result, N)
    recall = Evaluation.Recall(train, test, result, N)
    coverage = Evaluation.Coverage(train, test, result, N)
    # popularity += Evaluation.Popularity(train, test, result, N)

    # popularity /= iteration

    # 输出结果
    print('ItemCF precision = %f' % precision)
    print('ItemCF recall = %f' % recall)
    print('ItemCF coverage = %f' % coverage)


