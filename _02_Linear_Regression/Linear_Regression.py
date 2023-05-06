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
    def __init__(self):
        pass
    
    #梯度下降法迭代训练模型参数,x为特征数据，y为标签数据，a为学习率，epochs为迭代次数
    def fit(self,x,y):  
        #计算总数据量
        m=x.shape[0]
        #给x添加偏置项
        X = np.concatenate((np.ones((m,1)),x),axis=1)
        #计算总特征数
        n = X.shape[1]
        #初始化W的值,要变成矩阵形式
        W=np.mat(np.ones((n,1)))
        #X转为矩阵形式
        xMat = np.mat(X)
        #y转为矩阵形式，这步非常重要,且要是m x 1的维度格式
        yMat =np.mat(y.reshape(-1,1))
        #循环epochs次
        for i in range(1000):
            gradient = xMat.T*(xMat*W-yMat)/m + 0.01 * np.sign(W)
            W=W-0.01 * gradient
        return W
    def predict(self,x,w):  #这里的x也要加偏置，训练时x是什么维度的数据，预测也应该保持一样
        return np.dot(x,w)


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
    lasso_reg = Lasso( )
    w=lasso_reg.fit(x=X_train,y=y_train)
    X_train = np.concatenate((np.ones((404,1)),X_train ),axis=1)
     xMat = np.mat(X_train)
    result = lasso_reg.predict(XMat,w) # 进行预测
    return float(result)
