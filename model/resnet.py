from keras.models import Sequential
from keras.layers import merge, Flatten, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D, ZeroPadding1D, AveragePooling1D, Convolution1D
from keras.layers.core import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input, add
from keras.optimizers import SGD
from model.callback import RocAucMetricCallback, EarlyStopAverage


import model


class ResNet50(model.BaseModel):

    def __init__(self, input_shape, classes):
        super(ResNet50, self).__init__()
        self.model = self._structure(input_shape, classes=classes)

    def _identity_block(self, x, nb_filter, kernel_size=4):
        k1, k2, k3 = nb_filter
        out = Conv1D(k1, 1)(x)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Conv1D(k2, kernel_size, padding='same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Conv1D(k3, 1)(out)
        out = BatchNormalization()(out)

        out = add([out, x])
        out = Activation('relu')(out)
        return out

    def _conv_block(self, x, nb_filter, kernel_size=4):
        k1, k2, k3 = nb_filter

        out = Conv1D(k1, 1, padding='valid')(x)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Conv1D(k2, kernel_size, strides=3, padding='same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Conv1D(k3, 1, padding='valid')(out)
        out = BatchNormalization()(out)

        x = Conv1D(k3, 1, strides=3, padding='valid')(x)
        x = BatchNormalization()(x)

        out = add([out, x])
        out = Activation('relu')(out)
        return out

    def _structure(self, input_shape, classes, resize=True, drop=True, regular=True):
        inp = Input(shape=(5000, 8))
        out = ZeroPadding1D(3)(inp)
        out = Conv1D(64, 7, strides=2)(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = MaxPooling1D(3, strides=2)(out)

        out = self._conv_block(out, [64, 64, 256])
        out = self._identity_block(out, [64, 64, 256])
        out = self._identity_block(out, [64, 64, 256])

        out = self._conv_block(out, [128, 128, 512])
        out = self._identity_block(out, [128, 128, 512])
        out = self._identity_block(out, [128, 128, 512])
        out = self._identity_block(out, [128, 128, 512])

        out = self._conv_block(out, [256, 256, 1024])
        out = self._identity_block(out, [256, 256, 1024])
        out = self._identity_block(out, [256, 256, 1024])
        out = self._identity_block(out, [256, 256, 1024])
        out = self._identity_block(out, [256, 256, 1024])
        out = self._identity_block(out, [256, 256, 1024])

        out = self._conv_block(out, [512, 512, 2048])
        out = self._identity_block(out, [512, 512, 2048])
        out = self._identity_block(out, [512, 512, 2048])

        out = AveragePooling1D(7)(out)
        out = Flatten()(out)
        out = Dense(4096)(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Dropout(0.5)(out)
        out = Dense(4096)(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Dropout(0.5)(out)
        out = Dense(1000)(out)
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


class ResNetTest(model.BaseModel):

    def __init__(self, input_shape, classes):
        super(ResNetTest, self).__init__()
        self.model = self._structure(input_shape, classes=classes)

    def _identity_block(self, x, nb_filter, kernel_size=3):
        k1, k2, k3 = nb_filter
        out = Conv1D(k1, 1)(x)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Conv1D(k2, kernel_size, padding='same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Conv1D(k3, 1)(out)
        out = BatchNormalization()(out)

        out = add([out, x])
        out = Activation('relu')(out)
        return out

    def _conv_block(self, x, nb_filter, kernel_size=3, strides=1):
        k1, k2, k3 = nb_filter

        out = Conv1D(k1, 1, padding='valid')(x)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Conv1D(k2, kernel_size, strides=strides, padding='same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Conv1D(k3, 1, padding='valid')(out)
        out = BatchNormalization()(out)

        x = Conv1D(k3, 1, strides=strides, padding='valid')(x)
        x = BatchNormalization()(x)

        out = add([out, x])
        out = Activation('relu')(out)
        return out

    def _structure(self, input_shape, classes, resize=True, drop=True, regular=True):
        inp = Input(shape=input_shape)
        out = ZeroPadding1D(3)(inp)
        out = Conv1D(64, 7, strides=2)(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = MaxPooling1D(3, strides=2)(out)

        out = self._conv_block(out, [64, 64, 256])
        out = self._identity_block(out, [64, 64, 256])
        out = MaxPooling1D(3, strides=3)(out)

        out = self._conv_block(out, [128, 128, 512])
        out = self._identity_block(out, [128, 128, 512])
        out = MaxPooling1D(3, strides=3)(out)

        out = self._conv_block(out, [256, 256, 1024])
        out = self._identity_block(out, [256, 256, 1024])
        out = MaxPooling1D(3, strides=3)(out)

        out = self._conv_block(out, [512, 512, 2048])
        out = self._identity_block(out, [512, 512, 2048])
        out = MaxPooling1D(3, strides=3)(out)

        out = self._conv_block(out, [1028, 1028, 4096])
        out = self._identity_block(out, [1028, 1028, 4096])
        out = MaxPooling1D(3, strides=3)(out)

        # out = self._conv_block(out, [2048, 2048, 8192])
        # out = self._identity_block(out, [2048, 2048, 8192])

        out = AveragePooling1D(5)(out)
        out = Flatten()(out)

        if classes == 2:
            out = Dense(2, activation='sigmoid')(out)
            model = Model(inp, out)
            model.compile(loss='binary_crossentropy',
                          optimizer='adadelta',
                          metrics=['accuracy'])
        else:
            out = Dense(classes, activation='softmax')(out)
            model = Model(inp, out)
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
