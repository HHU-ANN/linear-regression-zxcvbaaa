# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(data):
    pass
    
def lasso(data):
    pass

def read_data(path='./data/exp02/'):#path data是上一级目录的
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

def main(data):
    x,y=read_data()
    weight=model(x.y)
    print(weight)
    return data @ weight

def model(x, y):
   def model(x, y):
    # 基础的最小二乘实现线性回归
    # $(X^TX)^-1(X^Ty)$
    lam = 0.2
    xTx=np.dot(x.T,x)
    denom=xTx+np.eye(np.shape(x)[1])*lam
    if np.linalg.det(denom)==0.0:
        print("矩阵为奇异矩阵，不能转置")
        return
    ws=denom.I*(x.T*y)
    return ws
