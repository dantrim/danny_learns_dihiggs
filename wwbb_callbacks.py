#!/usr/bin/env python

from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
import math

class roc_callback(Callback) :
    def on_train_begin(self, logs = {}) :
        self.aucs = []
        self.losses = []

    def on_train_end(self, logs = {}) :
        return

    def on_epoch_begin(self, epoch, logs = {}) :
        return

    def on_epoch_end(self, epoch, logs = {}) :
        self.losses.append(logs.get('val_loss'))
        y_pred = self.model.predict(self.validation_data[0])
        y_true = self.validation_data[1]
        auc = roc_auc_score(y_true, y_pred)
        print('roc_auc = %s' % str(round(auc,4)))
        self.aucs.append(auc)
        logs['roc_auc'] = self.aucs
        return

    def on_batch_begin(self, batch, logs = {}) :
        return

def lr_step_decay(epoch) : #initial_lr, drop, epoch_to_drop) :

    initial_lr = 0.5
    drop = 0.9
    epoch_to_drop = 3

    if epoch >= 50 :
        epoch == 50
    elif epoch >= 25 :
        epoch -= 25
#        new_lr = initial_lr
        print('INFO Setting learning rate back to initial LR (={})'.format(initial_lr))

    new_lr = initial_lr * math.pow(drop, math.floor((1+epoch)/epoch_to_drop))
    print('INFO LR Schedule: {}'.format(new_lr))
    return new_lr

#def wwbb_lr_schedule(initial_lr = lr, epoch_to_reset = 50) :
#
#    def schedule(epoch) :
#        if epoch == epoch_to_reset :
#            return initial_lr
#        else :
#            return 
#
