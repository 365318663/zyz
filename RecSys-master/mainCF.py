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
import datasplit
import LFM
import time
import ItemCF_hc
import numpy as np
import pandas as pd
from collections import Counter
import operator
import importlib
importlib.reload(UserCF)
importlib.reload(ItemCF)
importlib.reload(ItemCF_IUF)
importlib.reload(Evaluation)
importlib.reload(LFM)
importlib.reload(datasplit)
importlib.reload(ItemCF_hc)
import transE
importlib.reload(transE)



# 把数据读取成：用户id，电影id，评分=1的形式
# def readData():
#     data = []
#     fileName = './u.data'
#     fr = open(fileName,'r')
#     for line in fr.readlines():
#         lineArr = line.strip().split()
#         data.append([lineArr[0], lineArr[1], 1.0])
#     return data

    
# 随机分的，4份去train，1份去test
# def SplitData(data,M,k,seed):
#     test = []
#     train = []
#     random.seed(seed)
#     for user, item,rating in data:
#         if random.randint(0,M-1) == k:     # random.randint左闭右闭
#             test.append([user,item,rating])
#         else:
#             train.append([user, item,rating])
#     return train, test
        
    
# 将列表形式数据转换为dict形式
def transform(oriData):
    ret = dict()
    for user,item,rating in oriData:
        if user not in ret:
            ret[user] = dict()
        ret[user][item] = rating
    return ret  # { user:{ item:rating,},}
    
if __name__ == '__main__':
    # time_use = []
    start = time.clock()

    data, its = datasplit.readData('G:/WSPN/Knowledge Graph/beitaiping/SPO/data.txt')
    numFlod = 5
    precision =0
    recall = 0
    coverage = 0
    popularity =0
    precision_transE = 0
    recall_transE = 0
    coverage_transE = 0
    popularity_transE = 0

    W_transE = transE.ItemSimilarity(its)

    for i in range(0,numFlod):
        [oriTrain,oriTest] = datasplit.SplitData(data,4,i,0)
        # print('train:', len(oriTrain), 'test:', len(oriTest))
        proportion = len(oriTest) / ((len(oriTrain) + len(oriTest)) * 1.0)
        train = transform(oriTrain)
        test = transform(oriTest)
        # print('train:', len(train), 'test:', len(test))

        # W = UserCF.UserSimilarity(train)
    #    rank = UserCF.Recommend('1',train,W)
    #     result = UserCF.Recommendation(test.keys(), train, W)
    
        # W = UserCF_IIF.UserSimilarity(train)
    #    rank = UserCF_IIF.Recommend('1',train,W)
    #     result = UserCF_IIF.Recommendation(test.keys(), train, W)
        
        W = ItemCF_hc.ItemSimilarity(train,its)
    #    rank = ItemCF.Recommend('1',train,W)
        # 有个K参数，取与每个物品最相似的且不在该用户测试集里面的K个，K默认等于3
        result =  ItemCF_hc.Recommendation(test.keys(),train, W)


        #    rank = ItemCF.Recommend('1',train,W)
        # 有个K参数，取与每个物品最相似的且不在该用户测试集里面的K个，K默认等于3
        result_transE = transE.Recommendation(test.keys(), train, W_transE)

        # 查看，每个用户的候选推荐物品的数量
        # for i,l in result.items():
        #     print('推荐个数：',len(l))



#        W = ItemCF_IUF.ItemSimilarity(train)
    #    rank = ItemCF_IUF.Recommend('1',train,W)
#        result =  ItemCF_IUF.Recommendation(test.keys(),train, W)

#        [P,Q] = LFM.LatentFactorModel(train, 10,30, 0.02, 0.01)
#        rank = LFM.Recommend('2',train,P,Q)
#        result = LFM.Recommendation(test.keys(), train,P,Q)


        N = 5
        # 用户-物品训练集，用户-物品测试集，用户推荐字典集，推荐要N个
        precision += Evaluation.Precision(train,test, result,N)
        recall += Evaluation.Recall(train,test,result,N)
        coverage += Evaluation.Coverage(train, test, result,N)
        # popularity += Evaluation.Popularity(train, test, result,N)
        precision_transE += Evaluation.Precision(train, test, result_transE, N)
        recall_transE += Evaluation.Recall(train, test, result_transE, N)
        coverage_transE += Evaluation.Coverage(train, test, result_transE, N)
        print('========================')

    # 除以五次的平均
    precision /= numFlod
    recall /= numFlod
    coverage /= numFlod
    # popularity /= numFlod
    precision_transE /= numFlod
    recall_transE /= numFlod
    coverage_transE /= numFlod
    
     #输出结果
    print('precision = %f' %precision)
    print('recall = %f' %recall)
    print('coverage = %f' %coverage)
    # print('popularity = %f' %popularity)
    # print('proportion = %f' %proportion)
    print('precision_transE = %f' % precision_transE)
    print('recall_transE = %f' % recall_transE)
    print('coverage_transE = %f' % coverage_transE)

    # 保存结果

    # data = pd.DataFrame([[precision,recall,coverage,popularity,proportion]], columns=['precision','recall','coverage','popularity','proportion'])
    # data.to_csv('./result/result.csv')


    # time use
    # end = time.clock()
    # time_use = start - end
    # data = pd.DataFrame({'time_use': time_use}, index=[0])
    # data.to_csv('./result/time.csv')
    