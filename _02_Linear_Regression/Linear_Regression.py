# 最终在main函数中传入一个维度为6的numpy数组，输出预测值

import os

try:
    import numpy as np
except ImportError as e:
    os.system("sudo pip3 install numpy")
    import numpy as np

def ridge(x,y):
     lam = 0.2
    xTx=np.dot(x.T,x)
    denom=xTx+np.eye(np.shape(x)[1])*lam
    ws=denom.I*(x.T*y)
    return ws
    
#def lasso(data):
   # pass

def read_data(path='./data/exp02/'):#path data是上一级目录的
    x = np.load(path + 'X_train.npy')
    y = np.load(path + 'y_train.npy')
    return x, y

def main(data):
    x,y=read_data()
    weight=ridge(x.y)
    print(weight)
    return data @ weight

    
   
