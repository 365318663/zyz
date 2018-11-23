import tensorflow as tf
import argparse
import numpy as np
import os.path
import math
from collections import defaultdict
import operator


class TransE:
    @property
    def entity2id(self):
        return self.__entity2id

    @property
    def dimension(self):
        return self.__dimension


    @property
    def num_entity(self):
        return self.__num_entity



    def __init__(self, data_dir, negative_sampling, learning_rate,
                 batch_size, max_iter, margin, dimension, norm, evaluation_size, regularizer_weight):
        # this part for data prepare
        self.__data_dir = data_dir
        self.__negative_sampling = negative_sampling
        self.__regularizer_weight = regularizer_weight
        self.__norm = norm

        self.__entity2id = {}
        self.__id2entity = {}

        self.__num_entity = 0


        # load all the file: entity2id.txt, relation2id.txt, train.txt, test.txt, valid.txt
        self.load_data()
        print('finish preparing data. ')

        # this part for the model:
        self.__learning_rate = learning_rate
        self.__batch_size = batch_size
        self.__max_iter = max_iter
        self.__margin = margin
        self.__dimension = dimension
        # self.__norm = norm
        self.__evaluation_size = evaluation_size


    def load_data(self):
        print('loading entity2id.txt ...')
        with open(os.path.join(self.__data_dir, 'entity2id.txt'),'r',encoding='utf-8') as f:
            self.__entity2id = {line.strip().split(',')[0]: int(line.strip().split(',')[1]) for line in f.readlines()}
            self.__id2entity = {value: key for key, value in self.__entity2id.items()}


        self.__num_entity = len(self.__entity2id)


        print('entity number: ' + str(self.__num_entity))




def ItemSimilarity(items):
    parser = argparse.ArgumentParser(description="TransE")
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='the directory of dataset',
                        default='./zyzdata_demo_new/')
    parser.add_argument('--learning_rate', dest='learning_rate', type=float, help='learning rate', default=0.01)
    parser.add_argument('--batch_size', dest='batch_size', type=int, help="batch size", default=4096)
    parser.add_argument('--max_iter', dest='max_iter', type=int, help='maximum interation', default=100)
    parser.add_argument('--optimizer', dest='optimizer', type=str, help='optimizer', default='adam')
    parser.add_argument('--dimension', dest='dimension', type=int, help='embedding dimension', default=50)
    parser.add_argument('--margin', dest='margin', type=float, help='margin', default=1.0)
    parser.add_argument('--norm', dest='norm', type=str, help='L1 or L2 norm', default='L1')
    parser.add_argument('--evaluation_size', dest='evaluation_size', type=int, help='batchsize for evaluation',
                        default=500)
    parser.add_argument('--save_dir', dest='save_dir', type=str,
                        help='directory to save tensorflow checkpoint directory', default='output/')
    parser.add_argument('--negative_sampling', dest='negative_sampling', type=str,
                        help='choose unit or bern to generate negative examples', default='bern')
    parser.add_argument('--evaluate_per_iteration', dest='evaluate_per_iteration', type=int,
                        help='evaluate the training result per x iteration', default=10)
    parser.add_argument('--evaluate_worker', dest='evaluate_worker', type=int, help='number of evaluate workers',
                        default=4)
    parser.add_argument('--regularizer_weight', dest='regularizer_weight', type=float, help='regularization weight',
                        default=1e-5)
    parser.add_argument('--n_test', dest='n_test', type=int, help='number of triples for test during the training',
                        default=300)
    args = parser.parse_args()
    print(args)
    model = TransE(negative_sampling=args.negative_sampling, data_dir=args.data_dir,
                   learning_rate=args.learning_rate, batch_size=args.batch_size,
                   max_iter=args.max_iter, margin=args.margin,
                   dimension=args.dimension, norm=args.norm, evaluation_size=args.evaluation_size,
                   regularizer_weight=args.regularizer_weight)


    bound = 6 / math.sqrt(model.dimension)
    embedding_entity = tf.get_variable('embedding_entity', [model.num_entity, model.dimension],
                                              initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                        maxval=bound, seed=123))
    # embedding_relation = tf.get_variable('embedding_relation', [model.num_relation, model.dimension],
    #                                             initializer=tf.random_uniform_initializer(minval=-bound,
    #                                                                                       maxval=bound,
    #                                                                                       seed=124))


    saver = tf.train.Saver()
    W = dict()
    print(len(items))

    with tf.Session() as session:
        init_op = tf.global_variables_initializer()
        session.run(init_op)

        saver.restore(session, './model_new/transe')
        embedding_entity = session.run(embedding_entity)

        entity2id = model.entity2id


        for i in items:
            W.setdefault(i,{})
            for j in items:
                if i == j:
                    continue
                dist = sum((abs(embedding_entity[entity2id[i]]-embedding_entity[entity2id[j]])))
                W[i][j] = 1/(dist+1)
    return W


def Recommend(user_id, train, W):
    rank = dict()
    ru = train[user_id]
    for i, pi in ru.items():
        for j, wij in sorted(W[i].items(),key=operator.itemgetter(1), reverse=True):
            if j in ru:
                continue
            rank.setdefault(j, 0)
            rank[j] += pi * wij
    return rank


def Recommendation(users, train, W, K=3):
    result = dict()
    for user in users:
        rank = Recommend(user, train, W)
        R = sorted(rank.items(), key=operator.itemgetter(1),reverse=True)[0:K]
        result[user] = R
    return result

if __name__ == "__main__":
    ItemSimilarity()

