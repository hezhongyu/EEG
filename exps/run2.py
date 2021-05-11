# -*- coding: utf-8 -*-

# @Time    : 2018/11/24 0:58
# @Author  : Zhongyu
# @File    : run.py

from keras.utils import to_categorical
from keras import backend
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import time
import csv
import gc

from data.raw_data import RawData
from model.resnet import ResNetTest


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))


def run():
    # disease_list = [disease1, disease2, 'normal']
    disease_list = ['AF', 'BBB', 'TAC', 'normal']
    # disease_list = ['LVH', 'normal']
    raw_data = RawData(disease_list)
    raw_data.reshape_1d_2()
    # raw_data.to_12()
    raw_data.to_fft()

    # raw_data.split(2)
    data_count = raw_data.count_data_num()
    print('data shape:', raw_data.all_data[0][0].shape)
    print('data count:', data_count)
    gc.collect()

    exprm_num = 10
    result_list = []

    for i in range(exprm_num):
        print()
        print('experiment'+str(i))
        x_train, y_train, x_test, y_test = raw_data.get_train_test_data(0.8)
        y_train = to_categorical(y_train, num_classes=len(disease_list))
        y_test = to_categorical(y_test, num_classes=len(disease_list))
        # print(x_train)
        # print(y_train)
        # print(x_test)
        # print(y_test)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        # 建立新模型
        model = ResNetTest(input_shape=x_train[0].shape, classes=len(disease_list))
        # 加载旧模型
        # model = load_model('model/saved/pretrain/vgg19_normal_abnormal_0.h5')
        # print('train set:', model.evaluate(x_train, y_train, batch_size=128))
        # print('test set:', model.evaluate(x_test, y_test, batch_size=128))

        start = time.time()
        model.fit(x_train, y_train, batch_size=32, epochs=100)
        end = time.time()
        score1 = model.evaluate(x_train, y_train, batch_size=128)
        score2 = model.evaluate(x_test, y_test, batch_size=128)
        y_pred = model.predict(x_test, batch_size=128)

        print(score1)
        print(score2)
        # print(y_test)
        # print(y_pred)
        # for a in y_test:
        #     print(a[1])
        # for b in y_pred:
        #     print(b[1])
        print('time: ' + str(end-start))

        fpr, tpr, thresholds = roc_curve(y_test[:, 1], y_pred[:, 1])
        roc_auc = auc(fpr, tpr)
        print(roc_auc)

        # 存模型
        model_path = '../result/20210329/classes4_fft_' + str(i) + '.h5'
        model.save(model_path)

        # 存数据
        data_path = '../result/20210329/classes4_fft_' + str(i) + '.csv'
        with open(data_path, 'a+', newline='') as f:
            csv_write = csv.writer(f)
            csv_write.writerow(score1)
            csv_write.writerow(score2)
            csv_write.writerow(fpr)
            csv_write.writerow(tpr)
            csv_write.writerow(thresholds)
            csv_write.writerow([roc_auc])

        # plt.plot(fpr, tpr, lw=1, label='ROC of %s(area = %0.2f)' % (disease_list[1], roc_auc))
        # plt.show()

        result_list.append(score2[1])
        # 释放资源
        del(model)
        del(x_train, y_train, x_test, y_test)
        gc.collect()
        backend.clear_session()
        tf.reset_default_graph()

    data_path_s = '../result/20210329/classes4_fft.csv'
    with open(data_path_s, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(['data amount:', str(data_count)])
        csv_write.writerow(result_list)
        csv_write.writerow([sum(result_list)/len(result_list)])
        csv_write.writerow([max(result_list)])


if __name__ == '__main__':
    # out_disease_list = ['BRD', 'LVH', 'MI', 'PAC', 'PVC', 'TAC']
    # for i, _disease_1 in enumerate(out_disease_list):
    #     for _disease_2 in out_disease_list[i+1:]:
    #         if _disease_1 == 'BRD' and _disease_2 in ['LVH', 'MI', 'PAC', 'PVC']:
    #             continue
    #         else:
    #             # print(_disease_1, _disease_2)
    #             run(_disease_1, _disease_2)
    run()
