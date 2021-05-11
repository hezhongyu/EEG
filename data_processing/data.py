# -*- coding: utf-8 -*-

# @Time    : 2018/12/2 1:37
# @Author  : Zhongyu

import os
import random
import numpy as np
from numpy import fft, abs
import json
import mne


class Data(object):
    """
    生数据类，解决数据导入、处理等事务
    """

    MAIN_PATH = '../data/data_1500'

    def __init__(self):
        self.all_data_list = []


        self.disease_list = disease_list
        # 进行数据读取，先存列表再转numpy
        self._read_data()
        random.shuffle(self.all_data)
        # self.all_data = np.array(self.all_data_list)
        # print(self.all_data[0][0].shape)

    def load_data_from_xunkang_edf(self, path):
        _encephalitis_path = path + '/病毒性脑炎'
        _epilepsy_path = path + '/癫痫'
        for _file_name in os.listdir(_encephalitis_path):
            _file_path = '/'.join((_encephalitis_path, _file_name))
            raw = mne.io.read_raw_edf(data_path,
                                      eog=None,
                                      misc=None,
                                      stim_channel='auto',
                                      exclude=(),
                                      preload=False,
                                      verbose=None)

    # def _read_data(self):
    #     """
    #     从某一心律不齐文件夹中读取所有病例的数据
    #     :return:
    #     """
    #     for _disease_name in self.disease_list:
    #         print("read data: " + _disease_name)
    #         _disease_lable = self.disease_list.index(_disease_name)
    #         _sub_path = '/'.join((self.MAIN_PATH, _disease_name))
    #         for _file_name in os.listdir(_sub_path):
    #             _file_path = '/'.join((_sub_path, _file_name))
    #             if '.dat' not in _file_name:
    #                 continue
    #             else:
    #                 # 插入的是有两个元素元组， 第一个元素是心电数据，第二个数据是标签
    #                 self.all_data.append([self._read_data_from_file(_file_path), _disease_lable])
    #     return
    #
    # def _read_data_from_edf(self, file_name):
    #     """
    #     从文件中读取数据
    #     :param file_name: 文件名
    #     :return: 转化为二维矩阵输出（numpy格式），每一行向量为单导联的所有数据
    #     """
    #     _data = np.fromfile(file_name, dtype=np.int16)
    #     assert(len(_data) == 500*10*8)  # 500HZ, 10s, 8-lead
    #     # _data = _data.reshape([8, -1, 1])
    #     return _data

    def count_data_num(self):
        """
        计算数据量
        :return: 数据量
        """
        return len(self.all_data)

    def get_training_test(self, ptc, if_validation=False):
        """
        分离出训练集、测试集（、验证集）
        :param ptc: 若if_validation=False，参数为0-1之间的小数，表示训练集的比例；
                    若if_validation=True，参数为二元数组，第一位表示训练集的比例，第二位表示验证集的比例；
        :param if_validation: 是否需要分验证集
        :return: 若if_validation=False，返回 training_data, training_label, test_data, test_label
                若if_validation=True，返回 training_data, training_label, validation_data, validation_label, test_data, test_label
        """
        # 验证输入数据

        # 分离数据

        pass

        # assert 0 <= ptc <= 1
        # _all_nums = len(self.all_data)
        # _train_nums = int(_all_nums * ptc)
        # _test_nums = len(self.all_data) - _train_nums
        # # 为保证list与set转换中的问题，不对原数据进行选取，而是对序号进行选取
        # _train_index_list = random.sample(range(_all_nums), _train_nums)
        # _test_index_list = list(set(range(_all_nums)).difference(set(_train_index_list)))
        # _train_data = [self.all_data[i] for i in range(_all_nums) if i in _train_index_list]
        # _test_data = [self.all_data[i] for i in range(_all_nums) if i in _test_index_list]
        # _x_train = np.array([_train_data[i][0] for i in range(_train_nums)])
        # _y_train = np.array([_train_data[i][1] for i in range(_train_nums)])
        # _x_test = np.array([_test_data[i][0] for i in range(_test_nums)])
        # _y_test = np.array([_test_data[i][1] for i in range(_test_nums)])
        # return _x_train, _y_train, _x_test, _y_test


if __name__ == '__main__':
    path = '../data/xunkang'
    data = Data()
    data.load_data_from_xunkang_edf(path)

    import matplotlib.pyplot as plt

    # disease_list = ['normal', 'MI']
    # raw_data = RawData(disease_list)
    # raw_data.reshape_1d_2()
    # raw_data.first_sec_cut()
    # raw_data.get_train_test_data(0.8)

    path = 'C:/code/ECG/data/raw_data_file/normal'
    subpath = os.listdir(path)
    file = '/'.join((path, subpath[15]))
    _data = np.fromfile(file, dtype=np.int16)[:1000]
    plt.plot(_data)
    plt.show()
