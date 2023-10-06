# ！/user/bin/env python
# -*- coding:UTF-8 -*-
# author: yanglulu time:2022/10/22


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 根据log_iou修改行数
lines = 1736397
step = 5000
start_ite = 0
end_ite = 50200
igore = 1000
data_path = './my_coco3/log_iou.txt'  # log_loss的路径。
result_path = './my_coco3/Region Avg IOU'  # 保存结果的路径。

names = ['Region Avg IOU', 'Class', 'Obj', 'No Obj', '.5_Recall', '.7_Recall', 'count']
# result = pd.read_csv('log_iou.txt', skiprows=[x for x in range(lines) if (x%10==0 or x%10==9)]\
result = pd.read_csv(data_path, skiprows=[x for x in range(lines) if
                                          (x < lines * 1.0 / ((end_ite - start_ite) * 1.0) * igore or x % step != 0)] \
                     , error_bad_lines=False, names=names)
result.head()

for name in names:
    result[name] = result[name].str.split(': ').str.get(1)
result.head()
result.tail()
for name in names:
    result[name] = pd.to_numeric(result[name])
result.dtypes

####--------------
x_num = len(result['Region Avg IOU'].values)
tmp = (end_ite - start_ite - igore) / (x_num * 1.0)
x = []
for i in range(x_num):
    x.append(i * tmp + start_ite + igore)
# print(x)
print('total = %d\n' % x_num)
print('start = %d, end = %d\n' % (x[0], x[-1]))
####-------------


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, result['Region Avg IOU'].values, label='Region Avg IOU')
# ax.plot(result['Avg Recall'].values, label='Avg Recall')
plt.grid()
ax.legend(loc='best')
ax.set_title('The Region Avg IOU curves')
ax.set_xlabel('batches')
fig.savefig(result_path)