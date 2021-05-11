from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, BatchNormalization, Dropout
from keras.optimizers import SGD
from model.callback import RocAucMetricCallback, EarlyStopAverage

import model


class VGG16Model(model.BaseModel):
    """
    VGG16的cnn分类器
    """

    def __init__(self, input_shape, classes):
        super(VGG16Model, self).__init__()
        self.model = self._structure(input_shape, classes=classes)

    def _structure(self, input_shape, classes, resize=True, drop=True, regular=True):
        model = Sequential()

        # block1
        model.add(Conv1D(filters=64, kernel_size=4, padding='valid', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(filters=64, kernel_size=4, padding='valid', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=4))

        # block2
        model.add(Conv1D(128, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(128, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=4))

        # block3
        model.add(Conv1D(256, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(256, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(256, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=4))

        # block4
        model.add(Conv1D(512, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(512, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(512, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=4))

        # block5
        model.add(Conv1D(512, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(512, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(512, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=4))

        # 循环层结束
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000))
        model.add(BatchNormalization())
        model.add(Activation('relu'))


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


class VGG19Model(model.BaseModel):
    """
    VGG16的cnn分类器
    """

    def __init__(self, input_shape, classes):
        super(VGG19Model, self).__init__()
        self.model = self._structure(input_shape, classes=classes)

    def _structure(self, input_shape, classes, resize=True, drop=True, regular=True):
        model = Sequential()

        # block1
        model.add(Conv1D(filters=64, kernel_size=4, padding='valid', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(filters=64, kernel_size=4, padding='valid', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=3))

        # block2
        model.add(Conv1D(128, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(128, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=3))

        # block3
        model.add(Conv1D(256, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(256, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(256, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(256, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=3))

        # block4
        model.add(Conv1D(512, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(512, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(512, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(512, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=3))

        # block5
        model.add(Conv1D(512, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(512, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(512, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Conv1D(512, 4, padding='valid'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling1D(pool_size=3))

        # 循环层结束
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1000))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

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

    def fit(self, x_train, y_train, batch_size, epochs, early_stop=False, validation_split=0, shuffle=False, check_period=5):
        if early_stop:
            # early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
            self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                           callbacks=[RocAucMetricCallback(), EarlyStopAverage(check_period)],
                           validation_split=validation_split, shuffle=shuffle,)
        else:
            self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                           callbacks=[RocAucMetricCallback()],)
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


