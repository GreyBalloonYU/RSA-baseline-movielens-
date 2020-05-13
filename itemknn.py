#itemknn推荐算法
import math
import pandas as pd
import numpy as np
class Itemknn:
    __topK = None #所有用户各自的topK推荐列表
    __itemnums = 0  #物品数
    __usernums = 0  #用户数
    __K = 0 #top-K推荐
    #存放着每个物品最相似的n个物品的ID和相似度，因此是个三维ndarray，shape:(物品数, n, 2) n为n个最相似的物品
    #similar_knn[i][j][0]存放物品ID，similar_knn[i][j][1]存放物品ID对应的相似度
    __similar_knn = None  
    __topK = None #所有用户各自的topK推荐列表
    def __init__(self, itemnums, usernums, K):
        self.__itemnums = itemnums
        self.__usernums = usernums
        self.__K = K
        self.__topK = np.empty((usernums, K))
    def getSimilarTable(self, X):   #得到一个存放着item-item的相似性矩阵的表
        sim = np.empty((self.__itemnums, self.__itemnums))  #物品之间的相似度矩阵
        avg_user = X.groupby('userid')['rating'].mean().values  #先计算每个用户打分的平均值
        item_rate = []
        X.groupby('itemid').apply(itemgroupby, item_rate)   #以每个物品为组排一下序
        for i in range(self.__itemnums):
            for j in range(i+1):  #求主对角线以下的元素，因为这个矩阵是个对称矩阵
                #要找到同时对物品i和物品j都打过分的用户的集合
                print("计算物品%d与物品%d的相似度"%(i+1, j+1))
                u = list(set.intersection(set(item_rate[i]), set(item_rate[j])))
                i_list = X.loc[lambda df: df['itemid'] == i + 1]
                j_list = X.loc[lambda df: df['itemid'] == j + 1]
                ij, ii, jj = 0, 0, 0
                for k in u:
                    #先找到用户u对物品i和物品j的评分
                    rui = i_list.loc[lambda df: df['userid'] == k]['rating'].values
                    ruj = j_list.loc[lambda df: df['userid'] == k]['rating'].values
                    u_aver = avg_user[k-1]    #用户u的平均评分
                    ij += (rui - u_aver) * (ruj - u_aver)
                    ii += (rui - u_aver) ** 2
                    jj += (ruj - u_aver) ** 2
                if len(u) == 0:
                    sim[i, j] = 0    #没有一个用户同时给物品i和j打分，那么就简单粗暴地让相似度为0
                else:
                    sim[i, j] = ij / (math.sqrt(ii) * math.sqrt(jj))    #求相似度
        for i in range(self.__itemnums):
            for j in range(i+1, self.__itemnums):
                sim[i, j] = sim[j, i]   #对称矩阵
        #把sim写进表格里面
        tmp = pd.DataFrame(sim)
        tmp.to_csv('itemsimilartable.csv', index=False)
    def fit(self, X, cv, n):    #n为指定的最近邻的邻居数量的值，X为一个dataframe（共三列，userID,itemID,rating）
        self.__similar_knn = np.empty((self.__itemnums, n, 2))
        avg_user = X.groupby('userid')['rating'].mean().values  #计算每个用户打分的平均值
        sim = np.empty((self.__itemnums, self.__itemnums))  #物品之间的相似度矩阵
        #sim = pd.read_csv('itemsimilartable.csv', int, delimiter=',').values    #从表中读取物品相似度矩阵
        #求完相似度以后给每个物品都挑出n个最相似的物品(knn)
        for i in range(self.__itemnums):
            self.__similar_knn[i, :, 0] = np.argsort(-sim[i, :])[:n] + 1    #存放物品ID
            self.__similar_knn[i, :, 1] = -np.sort(-sim[i, :])[:n]  #存放物品ID对应的评分
        #对于每个用户，去计算他的几个评分最高的物品，作为topK推荐
        for i in range(self.__usernums):
            print("给用户%d做推荐"%(i+1))
            predict = np.empty((self.__itemnums, ))   #用户i对于所有物品的评分预测值
            for j in range(self.__itemnums):
                similar_sum, rate_sum = 0, 0
                for k in range(n):  #n为邻居数量的值
                    similar_item = self.__similar_knn[j, k, 0]  #物品ID
                    user_rate = X.loc[lambda df: df['userid'] == i+1].loc[lambda df: df['itemid'] == similar_item]['rating'].values
                    if len(user_rate) == 0: #如果在用户里没有找到该物品的评分，则跳过
                        continue
                    rate_sum += self.__similar_knn[j, k, 1] * user_rate
                    similar_sum += self.__similar_knn[j, k, 1]
                if(similar_sum == 0):
                    predict[j] = avg_user[i]    #如果用户i不含有物品j的n个最相似物品的评分，那么就直接给物品j赋上用户i的平均打分
                else:
                    predict[j] = rate_sum / similar_sum
            self.__topK[i, :] = np.argsort(-predict)[:self.__K] + 1 #从高往低排用户对物品的打分，选出前K个作为topK推荐

def itemgroupby(X, item_rate):
    item_rate.append(list(X.iloc[:, 0].values))
    return X