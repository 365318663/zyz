# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 13:09:26 2017

@author: lanlandetian
"""

import math
import operator


def ItemSimilarity(train,its):
    #calculate co-rated users between items
    #构建用户-物品表
    W =dict()
    for i in its:
        W.setdefault(i, {})
        for j in its:
            if i == j:
                continue
            W[i].setdefault(j, 0)

    C = dict()
    N = dict()
    for u,items in train.items():
        for i in items:
            N.setdefault(i,0)
            N[i] += 1
            C.setdefault(i,{})
            for j in items:
                if i == j:
                    continue
                C[i].setdefault(j, 0)
                C[i][j] += 1

    #calculate finial similarity matrix W
    for i,related_items in C.items():
        for j,cij in related_items.items():
            W[i][j] = cij / math.sqrt(N[i] * N[j])
    return W

    
def Recommend(user_id,train, W):
    rank = dict()
    ru = train[user_id]
    for i,pi in ru.items():
        for j,wij in sorted(W[i].items(), \
                           key = operator.itemgetter(1), reverse = True):
            if j in ru:
                continue
            rank.setdefault(j,0)
            rank[j] += pi * wij
    return rank

                           
def Recommendation(users, train, W, K = 3):
    result = dict()
    for user in users:
        rank = Recommend(user,train,W)
        R = sorted(rank.items(), key = operator.itemgetter(1), \
                   reverse = True)[0:K]
        result[user] = R
    return result