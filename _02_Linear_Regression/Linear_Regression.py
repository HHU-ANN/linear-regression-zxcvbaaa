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

def Cost_function(theta, X, y, lmbda):
    m, n = X.shape
    h = np.matmul(X, theta)
    J = (1/(2*m)) * np.sum(np.square(h-y)) + (lmbda/(2*m)) * np.sum(np.abs(theta))
    return J

def Gradient_descent(theta, X, y, alpha, lmbda, num_iters):
    m = X.shape[0]
    J_history = np.zeros(num_iters)
    
    for i in range(num_iters):
        h = np.matmul(X, theta)
        error = h - y
        reg_term = (lmbda/m) * np.sign(theta)
        theta = theta - (alpha/m) * (np.matmul(X.T, error) + reg_term)
        
        J_history[i] = Cost_function(theta, X, y, lmbda)

    return theta, J_history
        
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
    iterations = 1000
    theta, J_history = Gradient_descent(theta, X_train, y_train, alpha）
   
    return float(np.dot(X_train, theta))
