# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:04:07 2018

@author: lanlandetian
"""

import math
# 获得用户的推荐列表集合[物品id，评分]
def GetRecommendation(result, user, N = 5000):
    rank = result[user]     # 用户user的推荐列表
    ret = []
    if len(rank)  > N:     # ？
        for item,rating in rank:
            ret.append((item,rating))
    else:
        ret = rank
    return ret
# 召回率
# def Recall(train,test,result,N = 5000):
#     hit = 0    # 命中数目
#     all = 0    # 所有
#     for user in test.keys():
#         tu = test[user]
#         rank = GetRecommendation(result, user, N)
#         for item, pui in rank:
#             if item in tu:
#                 hit += 1
#         all += len(tu)
#     return hit / (all * 1.0)
def Recall(train,test,result,N = 5000):
    hit = 0    # 命中数目
    all = 0    # 所有
    a = []
    b = []

    for user in test.keys():
        tu = test[user]
        rank = GetRecommendation(result, user, N)
        i=0
        for item, pui in rank:
            if item in tu:
                hit += 1
                i += 1
        b.append(i)
        all += len(tu)
        a.append(len(tu))
    # print(a)
    # print('----------------------')
    # print(b)
    print(hit)
    print(all)
    print('----------------------')
    # print(b)
    return hit / (all * 1.0)
# 准确率
def Precision(train, test,result, N = 5000):
    hit = 0
    all = 0
    for user in test.keys():
        tu = test[user]
        rank = GetRecommendation(result,user,N)
        for item, pui in rank:
            if item in tu:
                hit += 1
        all += len(rank)
    print(hit)
    print(all)
    print('----------------------')
    return hit / (all * 1.0)
    
def Coverage(train, test, result, N = 5000):
    recommend_items = set()
    all_items = set()
    for user in train.keys():
        for item in train[user].keys():
            all_items.add(item)
            
    for user in test.keys():
        rank = GetRecommendation(result,user,N)
        for item , pui in rank:
            recommend_items.add(item)
    return len(recommend_items) / (len(all_items) * 1.0)
    
    
def Popularity(train, test, result, N = 5000):
    item_popularity = dict()
    for user, items in train.items():
        for item in items.keys():
            if item not in item_popularity:
                item_popularity[item] = 0
            item_popularity[item] += 1
    
    ret = 0
    n = 0
    for user in test.keys():
        rank = GetRecommendation(result,user,N)
        for item,pui in rank:
            ret += math.log(1 + item_popularity[item])
            n += 1
    ret /= n * 1.0
    return ret