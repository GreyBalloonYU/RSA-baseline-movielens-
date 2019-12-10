#itemknn推荐算法
import numpy as np
class Itemknn:
    __topK = None #所有用户各自的topK推荐列表
    __itemnums = 0  #物品数
    __usernums = 0  #用户数
    __K = 0 # top-K推荐
    def __init__(self, itemnums, usernums, K):
        self.__itemnums = itemnums
        self.__usernums = usernums
        self.__K = K
    #item_features是每个物品的特征向量，是稀疏向量
    #def fit(self, X, item_features):
