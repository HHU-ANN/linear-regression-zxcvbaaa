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



X_train, y_train = read_data()

import numpy as np
 
class RidgeRegression:
 
    def __init__(self, lambda_v=0.05):
        
        # 范数项的系数
        self.lambda_v = lambda_v
        
        # 模型参数w（训练时初始化）
        self.w = None
    
    def ridge(self, X=X_train, y=y_train):
        #'''岭回归算法'''
        n = X.shape
        I = np.identity(n)
        tmp = np.linalg.inv(np.matmul(X.T, X) + self.lambda_v*I)
        tmp = np.matmul(tmp, X.T)
        return np.matmul(tmp, y)
    
    def _preprocess_data_X(self, X=X_train):
        #'''数据预处理'''
        
        # 扩展X，添加x0列并设置为1
        m, n = X.shape
        X_ = np.empty((m, n+1))
        X_[:,0] = 1
        X_[:, 1:] = X
        
        return X_
    
    def train(self, X_train, y_train):
        #'''训练模型'''
        
        # 预处理X_train(添加x0列并设置为1)
        _X_train = self._preprocess_data_X(X_train)
        
        # 使用岭回归算法估算w
        self.w = self._ridge(_X_train, y_train)
        
    def predict(self, X=X_train):
        #'''预测'''
        # 预处理X_train(添加x0列并设置为1)
        _X = self._preprocess_data_X(X)
        return np.matmul(_X, self.w)

# 建立Lasso回归类
class LassoRegression:
    def __init__(self, alpha, learning_rate=0.01, max_iterations=1000):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        
    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        self.weights = np.zeros((n_features, 1))
        
        for i in range(self.max_iterations):
            y_pred = np.dot(X, self.weights)
            error = y - y_pred
            # LASSO的梯度下降中的L1函数的导数形式，即如果x>0取1否则-1
            self.weights += self.learning_rate * (np.dot(X.T, error) -
                                                   self.alpha * np.sign(self.weights))
    
    def predict(self, X):
        y_pred = np.dot(X, self.weights)
        return y_pred

# 进行岭回归

def lasso(data):
    lasso_reg = LassoRegression(alpha=0.1)
    lasso_reg.fit(X_train,y_train)
    data = np.reshape(data,(1,-1))
    result = lasso_reg.predict(data) # 进行预测
    return float(result)
