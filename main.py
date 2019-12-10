import numpy as np
import pandas as pd
from popular import Popular
from svdplus import Svdplus
from metric import hitK, ndcgK

#将数据集(u.data)根据时间序列重新划分训练集和测试集
#dataset = pd.read_csv('./ml-100k/u.data', names=['userid','itemid','rating','timestamp'], delimiter='\t')
#trainset = dataset.groupby('userid').apply(lambda x: x.sort_values('timestamp').iloc[:-1]).reset_index(drop=True)
#testset = dataset.groupby('userid').apply(lambda x: x.sort_values('timestamp').iloc[-1]).reset_index(drop=True)
#trainset.to_csv('traindata.csv', index=False)
#testset.to_csv('testdata.csv', index=False)

movies_info = pd.read_csv('./ml-100k/u.info', header=None, delimiter=' ')
movies_scores = pd.read_csv('traindata.csv', int, delimiter=',') #读取数据集，得到一个电影评分的dataframe
testdata = pd.read_csv('testdata.csv', int, delimiter=',').values[:,1].reshape(movies_info.iat[0,0], 1) #读入测试集
#movies_feature = pd.read_csv('./ml-100k/itemutf8.txt', delimiter='|')
#print(movies_feature.head())

#explicit_popular = Popular(movies_info.iat[1,0], movies_info.iat[0,0], 5, True) #初始化一个基于流行度推荐的模型(explicit feedback)，K设为5
#topK_for_users = explicit_popular.fit(movies_scores.values[:,1:3])    #所有用户各自的topK推荐列表
#ex_hit5 = hitK(topK_for_users, testdata) #计算hit@5
#ex_ndcg5 = ndcgK(topK_for_users, testdata)   #计算ndcg@5

#implicit_popular = Popular(movies_info.iat[1,0], movies_info.iat[0,0], 5, False) #初始化一个基于流行度推荐的模型(implicit feedback)，K设为5
#topK_for_users = implicit_popular.fit(movies_scores.values[:,1:3])    #所有用户各自的topK推荐列表
#im_hit5 = hitK(topK_for_users, testdata) 
#im_ndcg5 = ndcgK(topK_for_users, testdata)

#随机预测一波(baseline)
#topK_for_users = np.empty((movies_info.iat[0,0], 5))
#topK_for_users[:,:] = np.random.randint(1, movies_info.iat[1,0]+1, 5)
#rand_hit5 = hitK(topK_for_users, testdata) 
#rand_ndcg5 = ndcgK(topK_for_users, testdata)

#print("基于流行度推荐(explicit feedback)的hit@5为: %f"%(ex_hit5))
#print("基于流行度推荐(implicit feedback)的hit@5为: %f"%(im_hit5))
#print("随机推荐的hit@5为: %f"%(rand_hit5))
#print()
#print("基于流行度推荐(explicit feedback)的ndcg@5为: %f"%(ex_ndcg5))
#print("基于流行度推荐(implicit feedback)的ndcg@5为: %f"%(im_ndcg5))
#print("随机推荐的ndcg@5为: %f"%(rand_ndcg5))

#svdplus = Svdplus(movies_info.iat[1,0], movies_info.iat[0,0], 5, 10)
#svdplus.fit(movies_scores.iloc[:, :3], movies_scores.shape[0] * 30, 0.001, 0.001, 0.005, 0.015) #开始训练，遍历30遍训练集样本
#svdplus.drawLearningCurve() #绘制学习曲线
#topK_for_users = svdplus.predict()  #开始推荐
#svd_hit5 = hitK(topK_for_users, testdata)
#svd_ndcg5 = ndcgK(topK_for_users, testdata)
#print("基于svd++算法推荐的hit@5为: %f"%(svd_hit5))
#print("基于svd++算法推荐的ndcg@5为: %f"%(svd_ndcg5))