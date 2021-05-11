# -*- coding: utf-8 -*-

# @Time    : 2018/11/24 22:51
# @Author  : Zhongyu
# @File    : multi-simple.py


from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, BatchNormalization, Dropout
from keras import regularizers
from keras.optimizers import SGD
from keras.layers import LeakyReLU


from sklearn.metrics import roc_auc_score
from keras import callbacks

import model


class SimpleModel(model.BaseModel):
    """
    结构简单的cnn分类器
    """

    def __init__(self, input_shape, classes):
        super(SimpleModel, self).__init__()
        self.model = self._structure(input_shape, classes=classes)

    def _structure(self, input_shape, classes, resize=True, drop=True, regular=True):
        model = Sequential()
        # model.add(Conv2D(filters=16, kernel_size=(1, 8), padding='valid', input_shape=input_shape))
        model.add(Conv1D(filters=16, kernel_size=8, padding='valid', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Conv1D(16, 6, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(16, 6, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(Dropout(0.25))

        model.add(Conv1D(32, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(32, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=3))
        model.add(Dropout(0.25))

        model.add(Conv1D(64, 2, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(64, 2, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.5))

        # 循环层结束
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        if classes == 2:
            model.add(Dense(2, activation='sigmoid'))
            model.compile(loss='binary_crossentropy',
                          optimizer='adadelta',
                          metrics=['accuracy'])
        else:
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.add(Dense(classes, activation='softmax'))
            model.compile(optimizer=sgd,
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


class RocAucMetricCallback(callbacks.Callback):
    def __init__(self, predict_batch_size=1024, include_on_batch=False):
        super(RocAucMetricCallback, self).__init__()
        self.predict_batch_size = predict_batch_size
        self.include_on_batch = include_on_batch

    def on_train_begin(self, logs={}):
        if not ('roc_auc_val' in self.params['metrics']):
            self.params['metrics'].append('roc_auc_val')

    def on_train_end(self, logs={}):
        pass

