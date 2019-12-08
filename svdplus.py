#svd++推荐算法
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
class Svdplus:
    __avgrating = 0 #所有物品的平均打分(μ)
    __bu = None #用户偏置(bu)  shape=(用户数, )
    __bi = None #物品偏置(bi)  shape=(物品数, )
    __p = None #代表用户的向量  shape=(用户数, 因子数量)
    __q = None #代表物品的向量  shape=(物品数, 因子数量)
    __y = None #代表物品的向量(隐式反馈中的)  shape=(物品数, 因子数量)
    __nu = [] #每个用户的所有具有隐式反馈的物品ID，是一个二维数组(list)
    __topK = None #所有用户各自的topK推荐列表
    __K = 0 #top-K推荐
    __learn = [[],[]] #绘制学习曲线所用的矩阵，为一个二维数组
    __usernums = 0  #用户数
    __itemnums = 0  #物品数
    __factor_nums = 0   #向量的因子数
    def __init__(self, itemnums, usernums, K, factor_nums):
        self.__K = K
        self.__bu = np.random.rand(usernums, )  #随机初始化
        self.__bi = np.random.rand(itemnums, )
        self.__p = np.random.rand(usernums, factor_nums)
        self.__q = np.random.rand(itemnums, factor_nums)
        self.__y = np.random.rand(itemnums, factor_nums)
        self.__topK = np.empty((usernums, K))
        self.__usernums = usernums
        self.__itemnums = itemnums
        self.__factor_nums = factor_nums
    def drawLearningCurve(self): #绘制学习曲线的函数
        plt.plot(self.__learn[0], self.__learn[1])
        plt.xlabel("iterations")
        plt.ylabel("cost function")
        plt.title("learning curve")
        plt.show()
    #接收一个训练集，这个训练集(dataframe)有三列组成，第一列是用户的ID，第二列是电影的ID，第三列是该用户对这个电影的评分，返回一个top-K推荐列表
    #利用随机梯度下降
    def fit(self, X, iters, alpha1, alpha2, lamda1, lamda2): 
        #先根据X来给nu赋值
        X.groupby('userid').apply(nugroupby, self.__nu)
        X_train = X.values  #转成ndarray
        m = X_train.shape[0]    #训练集的数据个数
        np.random.shuffle(X_train)  #打乱训练集
        self.__avgrating = np.mean(X_train[:, -1])  #算一下平均rating
        avgcost_1000 = 0    #1000次迭代代价函数的平均值
        #开始迭代
        for i in range(iters):
            if i % 1000 == 999: 
                avgcost_1000 /= 1000
                self.__learn[0].append(int(i / 1000))
                self.__learn[1].append(avgcost_1000)
                print("已迭代1000次! 平均cost:%f"%(avgcost_1000))
                avgcost_1000 = 0
            j = i % m
            data = X_train[j, :]   #当前数据
            bu = self.__bu[data[0]-1]
            bi = self.__bi[data[1]-1]
            pu = self.__p[data[0]-1]
            nu = self.__nu[data[0]-1]
            yj = self.__y[nu, :]
            qi = self.__q[data[1]-1]
            bui = self.__avgrating + bu + bi
            yj_sum = 1 / np.sqrt(len(nu)) * np.sum(yj, axis=0)
            u_vec = pu + yj_sum #求一下该用户的向量
            predict = bui + np.matmul(qi, u_vec.reshape((self.__factor_nums, 1)))    #预测该商品的价格
            e = data[2] - predict   #预测值与真实值的偏差
            cost = e ** 2 + lamda1 * (bu ** 2 + bi ** 2) + lamda2 * (np.sum(qi ** 2) + np.sum(pu ** 2) + np.sum(yj ** 2))   #代价函数
            avgcost_1000 += cost
            #开始调整参数
            self.__bu[data[0]-1] += alpha1 * (e - lamda1 * bu)
            self.__bi[data[1]-1] += alpha1 * (e - lamda1 * bi)
            self.__q[data[1]-1] += alpha2 * (e * (pu + yj_sum) - lamda2 * qi)
            self.__p[data[0]-1] += alpha2 * (e * qi - lamda2 * pu)
            for j in nu:
                self.__y[j, :] += alpha2 * (e * 1 / np.sqrt(len(nu)) * qi - lamda2 * self.__y[j, :])
    #给每个用户topk推荐，推荐的电影不包括该用户在训练集中观看的电影
    def predict(self):
        #先算出每个用户未看过的电影ID
        allitems = list(np.arange(self.__itemnums))
        for i in range(self.__usernums):
            predictitems = np.array([var for var in allitems if var not in self.__nu[i]]) #用户未打分的物品ID集合
            prediction = np.vstack((predictitems, np.empty(predictitems.shape[0], ))).T
            bu = self.__bu[i]   #该用户的偏置
            pu = self.__p[i]    #该用户的向量
            yj_sum = 1 / np.sqrt(len(self.__nu[i])) * np.sum(self.__y[self.__nu[i], :], axis=0)    #该用户所有隐式反馈商品的向量
            for j in range(prediction.shape[0]):    #开始预测
                bi = self.__bi[int(prediction[j, 0])]   #物品的偏置
                qi = self.__q[int(prediction[j, 0])] #物品的向量
                prediction[j, 1] = self.__avgrating + bu + bi + np.matmul(qi, (pu + yj_sum).reshape(self.__factor_nums, 1)) #预测用户对该商品的评分
            #预测完以后要对这些电影的评分的排序作一个降序排列，然后取前K个评分最高的电影作为该用户的topK推荐
            tmp = pd.DataFrame(prediction, columns=['itemid', 'rating'])    #把ndarray转成dataframe
            tmp.sort_values('rating', ascending=False, inplace=True)    #根据评分排序
            self.__topK[i, :] = tmp.iloc[:self.__K, 0].values.astype(int)   #选出前K个评分最高的作为该用户的topK推荐
            print("预测完用户%d的topk"%(i + 1))
        return self.__topK

def nugroupby(X, nu):
    nu.append(list(X.iloc[:, 1].values - 1))    #由于python索引是从0开始，所以对应物品在数组中的值是它的ID减去1
    return X