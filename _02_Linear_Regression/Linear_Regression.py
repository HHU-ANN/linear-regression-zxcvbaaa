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

def gradient_descent(X, y, theta, alpha, num_iters, lambd):
    m = len(y)
    for i in range(num_iters):
        h = X.dot(theta)
        theta = theta - alpha * (1/m) * (X.T.dot(h-y) + lambd*np.sign(theta))
    return theta
     
        
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
    m, n = X_train.shape
    X = np.hstack((np.ones((m, 1)), X_train))
    theta = np.zeros(n+1)
    alpha = 0.01
    num_iters = 1000
    lambd = 0.1
    theta = gradient_descent(X, y_train, theta, alpha, num_iters, lambd)
    print(theta)
    return np.dot(X, theta)[0][0]
