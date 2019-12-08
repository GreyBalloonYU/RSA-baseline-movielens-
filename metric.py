import numpy as np

#评价指标: hit@K
#传入一个针对每个用户的topK推荐列表(ndarray)，shape为(用户数量，K)
#传入一个测试集合(ndarray)，代表每个用户实际上购买的商品，shape为(用户数量，某用户购买的商品数量)
def hitK(predictions, test):
    hits = 0 #topK推荐列表命中测试集的数据
    for i in range(test.shape[0]):
        for j in range(test.shape[1]):
            hits += np.sum(predictions[i] == test[i,j])
    return hits / test.shape[0] #test.shape[0]为用户总数，作为hit ratio的分母

#评价指标: ndcg@K
#传入一个针对每个用户的topK推荐列表(ndarray)，shape为(用户数量，K)
#传入一个测试集合(ndarray)，代表每个用户实际上购买的商品，shape为(用户数量，某用户购买的商品数量=1)，暂时只考虑测试集里每个用户只有一个商品的情况
#传入一个电影特征的相关性矩阵r
#因为每个用户的ndcg都不一样，所以这个函数返回的是所有用户ndcg@K的平均值
def ndcgK(predictions, test):
    total_ndcg = 0
    K = predictions.shape[1]
    for i in range(test.shape[0]):  #遍历每个用户的测试集
        for j in range(K): #遍历topK推荐的商品
            total_ndcg += np.sum(predictions[i,j] == test[i]) / np.log2(j + 2)
    return total_ndcg / test.shape[0]   #返回所有用户ndcg@K的平均值