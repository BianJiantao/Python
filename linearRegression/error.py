import csv
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import matplotlib.pyplot as plt

# 1. 读取预测数据
n_row = 0
predict = []
text = open('result/predict.csv', 'r', encoding='big5')
row = csv.reader(text , delimiter=",")
for r in row:
    # 第0行没有数据
    if n_row != 0:
        predict.append(float(r[1]))
    n_row = n_row+1
text.close()

# 2. 读取实际结果
n_row = 0
ans = []
text2 = open('result/ans.csv', 'r', encoding='big5')
row2 = csv.reader(text2 , delimiter=",")
for r2 in row2:
    # 第0行没有数据
    if n_row != 0:
        ans.append(float(r2[1]))
    n_row = n_row+1
text2.close()

# print(predict)
# print(ans)
ans = np.array(ans)
predict =  np.array(predict)

loss = ans - predict
cost = np.sum(loss ** 2) / len(ans)
cost_a = math.sqrt(cost)
print(len(ans),cost,cost_a)

x = range(240)
plt.figure()
plt.plot(x,loss)
plt.savefig("error.png")