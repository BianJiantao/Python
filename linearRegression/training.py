import csv
import numpy as np
from numpy.linalg import inv
import random
import math
import sys

data = []
# 每一个维度存储一种污染物的信息
for i in range(18):
	data.append([])

# 1. 读取数据
n_row = 0
text = open('data/train.csv', 'r', encoding='big5')
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0行没有数据
    if n_row != 0:
        # 每一列只有第3-27格有值(1天內24小時的數值)
        for i in range(3,27):
            if r[i] != "NR":
                data[(n_row-1)%18].append(float(r[i]))
            else:
                data[(n_row-1)%18].append(float(0))
    n_row = n_row+1
text.close()

# 2.解析数据
x = []
y = []
# 每 12 個月
for i in range(12):
    # 一個月取連續10小時的data可以有471筆
    for j in range(471):
        x.append([])
        # 18種污染物
        for t in range(18):
            # 連續9小時
            for s in range(9):
                x[471*i+j].append(data[t][480*i+j+s] )
        y.append(data[9][480*i+j+9])
x = np.array(x)
y = np.array(y)

# print(x.shape[0])
# print( np.ones((x.shape[0],1)) )

# add square term
# x = np.concatenate((x,x**2), axis=1)

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)
# print(x.shape)

# 3. init weight & other hyperparams
w = np.zeros(len(x[0]))
l_rate = 1
repeat = 10000
lamda = 0.1

# 4. start training
x_t = x.transpose()
s_gra = np.zeros(len(x[0]))
# print(len(x))

for i in range(repeat):
    hypo = np.dot(x,w)
    loss = hypo - y
    cost = (np.sum(loss**2) + lamda*(np.sum(w**2)) ) / len(x)
    cost_a  = math.sqrt(cost)
    gra = 2*np.dot(x_t,loss) + 2*lamda*w
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    if i == repeat-1:
        print ('iteration: %d | Cost: %f  ' % ( i,cost_a))


#  * check your ans with close form solution
# use close form to check whether ur gradient descent is good
# however, this cannot be used in hw1.sh
w2 = np.matmul(np.matmul(inv(np.matmul(x.transpose(),x)),x.transpose()),y)
print("w2-w = ",w-w2)

# 5. save/read model
# save model
np.save('model.npy',w2)
# print(w)