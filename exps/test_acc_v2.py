# -*- coding: utf-8 -*-

# @Time    : 2019/8/27 10:28
# @Author  : Zhongyu
# @File    : test_acc.py


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model
import csv
from numpy import abs, fft


DISEASE = ['AF', 'normal']
disease_name = 'AF'

file_path = 'data/test/多元模型验证数据12000/' + disease_name+'3000'
all_data = []
file_name_list = []
for _file_name in os.listdir(file_path):
    if '.dat' not in _file_name:
        continue
    else:
        final_path = '/'.join((file_path, _file_name))
        _data = np.fromfile(final_path, dtype=np.int16)
        assert(len(_data) == 500*10*8)  # 500HZ, 10s, 8-lead
        all_data.append(_data)
        file_name_list.append(_file_name)

temp_data_list = []
for line in all_data:
    line = np.r_[line[:10000], line[15000:20000]].reshape([-1, 3])
    temp_data_list.append(line)

x_test = np.array([temp_data_list[i] for i in range(len(temp_data_list))])
# print(x_test[0])
# exit(1)

model = load_model('model20200424/classes4_0.h5')
model.summary()

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
y_pred = model.predict(x_test, batch_size=32)
print(y_pred)

data_path = 'data/test/result/20200428/'+disease_name+'_result.csv'
with open(data_path, 'a+', newline='') as f:
    csv_write = csv.writer(f)
    for i in range(len(file_name_list)):
        csv_write.writerow([file_name_list[i], y_pred[i][0], y_pred[i][1], y_pred[i][2], y_pred[i][3]])

