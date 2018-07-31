import keras
from sklearn.metrics import roc_auc_score

class AUCHistory(keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
        self.aucs = []
    
    def on_train_begin(self, logs={}):
        self.aucs = []

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.X_val)
        self.aucs.append(roc_auc_score(self.y_val[:,0], y_pred[:,0]))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return