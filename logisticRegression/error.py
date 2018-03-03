import csv
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import matplotlib.pyplot as plt
import pandas as pd


# 1. 读取预测数据
predict = pd.read_csv('logistic_output/log_prediction.csv', sep=',', header=0)
predict = np.array(predict.values)

# 2. 读取实际结果
ans = pd.read_csv('logistic_output/correct_answer.csv', sep=',', header=0)
ans = np.array(ans.values)

# print(predict)
# print(ans)

loss = ans - predict
cost = np.sum(loss ** 2) / len(ans)
cost_a = math.sqrt(cost)
print(len(ans),cost,cost_a)

x = range(len(ans))
plt.figure()
plt.plot(x,loss)
plt.savefig("error.png")