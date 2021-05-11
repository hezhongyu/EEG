from keras import callbacks
from keras.callbacks import EarlyStopping


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


class EarlyStopAverage(callbacks.Callback):
    def __init__(self, check_period):
        super(EarlyStopAverage, self).__init__()
        self.check_period = check_period

    def on_train_begin(self, logs={}):
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        if epoch >= 2*self.check_period and epoch % self.check_period == 0:
            print('average val_loss: ', sum(self.val_losses[-self.check_period:])/self.check_period)
            if sum(self.val_losses[-self.check_period:])/self.check_period > sum(self.val_losses[-2*self.check_period:-self.check_period])/self.check_period:
                self.model.stop_training = True

