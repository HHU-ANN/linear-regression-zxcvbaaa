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

class standard_ridge():
    def __init__(self):
        pass

    def fit(self,x=X_train,y=y_train,Lambda):
        m = x.shape[0]
        X = np.concatenate((np.ones((m,1)),x),axis=1)
        xMat= np.mat(X)
        yMat = np.mat(y.reshape(-1,1))
        xTx = xMat.T * xMat
        rxTx = xTx + np.eye(xMat.shape[1]) * Lambda * m
        #rxTx.I为rxTx的逆矩阵
        w = rxTx.I * xMat.T * yMat
        return w

def lasso(data):
    pass


