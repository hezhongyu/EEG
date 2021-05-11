# -*- coding: utf-8 -*-

# @Time    : 2020/8/4 18:06
# @Author  : Zhongyu
# @File    : run_10_fold.py


"""
规定好分类的10折交叉验证训练
"""

import csv
import gc
import time

from keras.utils import to_categorical
from keras.backend.tensorflow_backend import set_session
from keras import backend
import tensorflow as tf
from sklearn.metrics import roc_curve, auc

from data.raw_data import RawDataFromFile
from model.resnet import ResNetTest


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


def run():
    disease_list = ['MI', 'normal']
    # 读取分割文件
    segment_file_list = []
    all_file_list = []
    with open('file_segment.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            segment_file_list.append(row)
            all_file_list.extend(row)
    # 按照十折分为训练集与测试集
    result_list = []
    for i in range(len(segment_file_list)):
        test_file_list = segment_file_list[i]
        train_file_list = list(set(all_file_list).difference(set(test_file_list)))

        # 按照文件名加载数据
        raw_data = RawDataFromFile(disease_list, train_file_list, test_file_list)
        raw_data.reshape_1d_2()
        raw_data.to_fft()
        data_count = raw_data.count_data_num()
        print("train data count: ", data_count[0])
        print("train data count: ", data_count[1])
        x_train, y_train, x_test, y_test = raw_data.get_train_test_data()
        y_train = to_categorical(y_train, num_classes=len(disease_list))
        y_test = to_categorical(y_test, num_classes=len(disease_list))
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        # 进行训练
        print('experiment', i)
        model = ResNetTest(input_shape=x_train[0].shape, classes=len(disease_list))
        start = time.time()
        model.fit(x_train, y_train, batch_size=32, epochs=100)
        end = time.time()
        score1 = model.evaluate(x_train, y_train, batch_size=128)
        score2 = model.evaluate(x_test, y_test, batch_size=128)
        y_pred = model.predict(x_test, batch_size=128)
        print(score1)
        print(score2)
        print('time: ' + str(end-start))

        fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred[:, 1])
        roc_auc = auc(fpr, tpr)
        print(roc_auc)

        # 存模型
        model_path = 'save/test_20200804/MI_frequency_' + str(i) + '.h5'
        model.save(model_path)
        # 存数据
        data_path = 'save/test_20200804/MI_frequency_' + str(i) + '.csv'
        with open(data_path, 'a+', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(score1)
            csv_write.writerow(score2)
            csv_write.writerow(fpr)
            csv_write.writerow(tpr)
            csv_write.writerow(thresholds)
            csv_write.writerow([roc_auc])

        result_list.append(score2[1])
        # 释放资源
        del(model)
        del(x_train, y_train, x_test, y_test)
        gc.collect()
        backend.clear_session()
        tf.reset_default_graph()

    data_path_s = 'save/test_20200804/MI_frequency.csv'
    with open(data_path_s, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(['data amount:', str(len(all_file_list))])
        csv_write.writerow(result_list)
        csv_write.writerow([sum(result_list) / len(result_list)])
        csv_write.writerow([max(result_list)])


if __name__ == '__main__':
    run()
