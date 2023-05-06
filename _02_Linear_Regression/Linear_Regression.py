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

# 建立Lasso回归类
class Lasso():
    def __init__(self, alpha=1, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
    
    #梯度下降法迭代训练模型参数,x为特征数据，y为标签数据，a为学习率，epochs为迭代次数
    def fit(self,X,y):  
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)

        for i in range(self.max_iter):
            grad = self._compute_gradient(X, y)
            self.coef_ -= self.alpha * grad
            self.coef_ = self._soft_threshold(self.coef_, self.alpha)
            if np.linalg.norm(grad, ord=1) < self.tol:
                break
           
    def predict(self,X):  #这里的x也要加偏置，训练时x是什么维度的数据，预测也应该保持一样
         return np.dot(X, self.coef_) 
        
    def _compute_gradient(self, X, y):
        y_pred = self.predict(X)
        error = y_pred - y
        grad = np.dot(X.T, error)
        return grad
       
    def _soft_threshold(self, coef, alpha):
         return np.sign(coef) * np.maximum(np.abs(coef) - alpha, 0)
        
        
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
    lasso_reg = Lasso(alpha=0.1, max_iter=1000, tol=1e-6)
    lasso_reg.fit(X_train,y_train)
    y_pred = lasso_reg.predict(X_train)# 进行预测
    return y_pred
