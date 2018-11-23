# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 13:09:26 2017

@author: lanlandetian
"""

import math
import operator


def ItemSimilarity(train):
    #calculate co-rated users between items
    #构建物品相似性表，返回仍是二维字典
    C =dict()
    N = dict()
    for u,items in train.items(): # 返回列表，每个单元表示成元祖（key，values）
        for i in items:     # 读取的key值
            N.setdefault(i,0)  #没有i键，其value设为0，有i键，其value保持
            N[i] += 1     # N保存用品i由多少用户用到
            C.setdefault(i,{})    # C保存物品i和j同时由多少用户用到
            for j in items:
                if i == j:
                    continue
                C[i].setdefault(j,0)
                C[i][j] += 1

    #calculate finial similarity matrix W
    W = C.copy()
    for i,related_items in C.items():
        for j,cij in related_items.items():
            W[i][j] = cij / math.sqrt(N[i] * N[j])
    return W

# 测试集中一个用户的id，训练集，物品关联矩阵，推荐个数
# 返回特定user_id用户推荐的物品字典表，{物品id：推荐分}
# def Recommend(user_id,train, W,K = 3):
#     rank = dict()
#     ru = train[user_id]
#     # i是user_id买过的物品id，pi是评分（这里全是1）
#     for i,pi in ru.items():
#         # 得到与i最相似的k个物品，返回[(key,values),(),()……]
#         for j,wij in sorted(W[i].items(), \
#                            key = operator.itemgetter(1), reverse = True)[0:K]:
#             # j和ru的key比较
#             if j in ru:
#                 continue
#             rank.setdefault(j,0)
#             rank[j] += pi * wij
#     return rank   # {物品id：评分}

# 原来算法，如果和j相似的多数在ru里面，就会造成j的数量急剧下降
# 解决方法，引入物品集合（大动），取消k（改变书中算法）
def Recommend(user_id,train, W,K = 3):
    rank = dict()
    ru = train[user_id]
    # i是user_id买过的物品id，pi是评分（这里全是1）
    for i,pi in ru.items():
        # 得到与i最相似的k个物品，返回[(物品id,平分),(),()……]
        for j,wij in sorted(W[i].items(), \
                           key = operator.itemgetter(1), reverse = True)[0:K]:
            # j和ru的key比较
            if j in ru:
                continue
            rank.setdefault(j,0)
            rank[j] += pi * wij
    return rank   # {物品id：评分}
    
    
#class Node:
#    def __init__(self):
#        self.weight = 0
#        self.reason = dict()
#    
#def Recommend(user_id,train, W,K =3):
#    rank = dict()
#    ru = train[user_id]
#    for i,pi in ru.items():
#        for j,wij in sorted(W[i].items(), \
#                           key = operator.itemgetter(1), reverse = True)[0:K]:
#            if j in ru:
#                continue
#            if j not in rank:
#                rank[j] = Node()
#            rank[j].reason.setdefault(i,0)
#            rank[j].weight += pi * wij
#            rank[j].reason[i] = pi * wij
#    return rank

# test集的user_id,train数据，物品关联表
# 返回用户-推荐字典表,{user_id:[(物品id，推荐分),(),()……],……}，对每个用户，倒叙排列
def Recommendation(users, train, W, K = 3):
    result = dict()
    for user in users:
        rank = Recommend(user,train,W,K)
        # print("某个用户的推荐数",len(rank))
        R = sorted(rank.items(), key = operator.itemgetter(1), \
                   reverse = True)   # 返回list，元祖（key，value）
        # 推荐物品倒序排列，
        result[user] = R
    return result   # {userid:[(物品id，推荐分数)],}