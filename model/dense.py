# -*- coding: utf-8 -*-

# @Time    : 2020/8/5 18:17
# @Author  : Zhongyu
# @File    : dense.py


from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization, Dropout
from keras.optimizers import SGD
from keras.layers import Input
from keras.models import Model

import model
from model.callback import RocAucMetricCallback


class DenseModel(model.BaseModel):
    """
    全连接分类器
    """

    def __init__(self, input_shape, classes):
        super(DenseModel, self).__init__()
        self.model = self._structure(input_shape, classes=classes)

    def _structure(self, input_shape, classes, resize=True, drop=True, regular=True):
        model = Sequential()
        inp = Input(shape=(8192,))

        out = Dense(8192)(inp)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Dropout(0.5)(out)
        out = Dense(8192)(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Dropout(0.5)(out)
        out = Dense(8192)(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        if classes == 2:
            out = Dense(2, activation='sigmoid')(out)
            model = Model(inp, out)
            model.compile(loss='binary_crossentropy',
                          optimizer='adadelta',
                          metrics=['accuracy'])
        else:
            out = Dense(classes, activation='softmax')(out)
            model = Model(inp, out)
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(optimizer='adadelta',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        return model

    def fit(self, x_train, y_train, batch_size, epochs):
        self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[RocAucMetricCallback()])
        return

    def evaluate(self, x_test, y_test, batch_size):
        score = self.model.evaluate(x_test, y_test, batch_size=batch_size)
        return score

    def predict(self, x_test, batch_size):
        y_predict = self.model.predict(x_test, batch_size)
        return y_predict

    def save(self, file_path):
        self.model.save(file_path)
        return
