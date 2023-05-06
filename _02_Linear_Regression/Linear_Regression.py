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

class Lasso:
    def __init__(self, alpha=0.1, tol=0.01, max_iter=1000, learning_rate=0.01):
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        
    def fit(self, X, y):
        n, m = X.shape
        self.theta = np.zeros(m)
        self.intercept = 0
        
        for iter_num in range(self.max_iter):
            grad = X.T.dot(X.dot(self.theta) + self.intercept - y) / n
            self.intercept -= self.learning_rate * grad[-1]
            self.theta -= self.learning_rate * (grad[:-1] + self.alpha * np.sign(self.theta))
            if np.max(np.abs(grad)) < self.tol:
                    break
     
    def predict(self, X):
        return X.dot(self.theta) + self.intercept
    
     
        
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
    lasso = Lasso(alpha=0.1, tol=0.01, max_iter=1000, learning_rate=0.01)
    lasso.fit(X_train, y_train)
    y_pred = lasso.predict(X_train)
    return float(y_pred)
