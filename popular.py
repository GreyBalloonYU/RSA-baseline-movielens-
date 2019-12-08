#基于流行度的推荐算法
import numpy as np
class Popular:
    __total_scores = None    #所有电影的评分之和，total_scores[i]表示所有用户对电影i的评分的总和
    __most_popular_movie = -1 #最流行的电影
    __topK = None #所有用户各自的topK推荐列表
    __K = 0 # top-K推荐
    __is_explicit = False #是否利用显式反馈(关注rating具体数值)，是---True，否---False
    def __init__(self, movienums, usernums, K, is_explicit):
        self.__total_scores = np.zeros((movienums+1, ))
        self.__K = K
        self.__most_popular_movie = np.empty((K, ))
        self.__topK = np.empty((usernums, K))
        self.__is_explicit = is_explicit
    def fit(self, X): #接收一个训练集，这个训练集(ndarray)有两列组成，第一列是电影的ID，第二列是某用户对这个电影的评分，返回一个top-K推荐列表
        if self.__is_explicit:
            #explicit feedback: 根据训练集求出所有电影的评分之和
            for i in range(X.shape[0]):
                self.__total_scores[X[i,0]] += X[i,1]
        else:
            #implicit feedback: 根据训练集求出所有电影的观看数之和
            for i in range(X.shape[0]):
                self.__total_scores[X[i,0]] += 1
        self.__most_popular_movie = np.argsort(-self.__total_scores)[:self.__K] #一个top-K推荐列表，越在前面(索引更小)的电影越受欢迎
        self.__topK[:,:] = self.__most_popular_movie
        return self.__topK   #返回所有用户各自的topK推荐列表