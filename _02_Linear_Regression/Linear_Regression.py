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

def cost(theta, X, y):
    residuals = y - X.dot(theta)
    rss = residuals.T.dot(residuals)  # 残差平方和
    return rss[0, 0]

def gradient(theta, X, y):
    residuals = y - X.dot(theta)
    grad = -X.T.dot(residuals) + L1_penalty*np.sign(theta)
    return grad
     
        
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
    m, n = X.shape
    theta = np.zeros(n).reshape(-1, 1)
    alpha = 0.01         # 学习率
    iters = 1000         # 迭代次数
    L1_penalty = 1   
    
    for i in range(iters):
    theta -= alpha * gradient(theta, X, y)
    
    return float(np.dot(X, theta))
