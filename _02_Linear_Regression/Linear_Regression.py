# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

import numpy as np

def read_data(path='./data/exp02/'):
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

# 建立岭回归类
class RidgeRegression:
    def __init__(self, alpha):
        self.alpha = alpha
    
    def fit(self, X, y):
        n_features = np.shape(X)[1]
        # 添加正则项，防止过拟合
        self.weights = np.dot(np.linalg.inv(np.dot(X.T, X) + 
                               self.alpha*np.eye(n_features)), 
                               np.dot(X.T, y))
    
    def predict(self, X):
        y_pred = np.dot(X, self.weights)
        return y_pred

def stageWise(xMat, yMat, eps=0.01, numIt=100):
    
    # 将数据转换为矩阵，并进行标准化处理
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    # 特征标准化
    xMat = regularize(xMat)
    m, n = shape(xMat)
    # 每次迭代的权重
    returnMat = zeros((numIt, n))
    # 创建向量ws来保存w的值
    ws = zeros((n, 1))
    wsBest = ws.copy()
    for i in range(numIt):  # 遍历每轮迭代
        print(ws.T)  # 打印w向量，用于分析执行的过程和效果
        # 设置当前最小误差
        lowestError = inf
        # 遍历每个特征，分别计算增加或减少该特征对误差的影响
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsBest = wsTest
        ws = wsBest.copy()
        returnMat[i, :] = ws.T
    return returnMat

        
        
# 进行岭回归
def ridge(data):
    X_train, y_train = read_data()
    ridge_reg = RidgeRegression(alpha=0.1) # 设置参数alpha
    ridge_reg.fit(X_train, y_train) # 使用训练数据拟合模型
    
    data = np.reshape(data, (1, -1)) # 将数据改为2D矩阵形式
    result = ridge_reg.predict(data) # 进行预测
    return float(result)
def lasso(data):
    X_train, y_train = read_data()
    return stageWise(X_train, y_train,0.01,200)
