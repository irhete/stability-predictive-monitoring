# Based on https://github.com/verenich/ProcessSequencePrediction

import time
from keras.models import Sequential, Model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers import Input
from keras.optimizers import Nadam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
import csv
from sklearn.metrics import mean_absolute_error, accuracy_score
import os
import sys
from dataset_manager import DatasetManager
from sys import argv

import pandas as pd
import numpy as np


dataset_name = argv[1]
cls_method = "lstm_singletask"

train_ratio = 0.8
val_ratio = 0.2

lstmsize = int(argv[2])
dropout = float(argv[3])
n_layers = int(argv[4])
batch_size = int(argv[5])
learning_rate = float(argv[6])
activation = argv[7]
optimizer = argv[8]

nb_epoch = 30

data_split_type = "temporal"
normalize_over = "train"

output_dir = "results"
#params = "pd_fixed_trainratio80_outcome_all_data_singletask"
params = "lstmsize%s_dropout%s_nlayers%s_batchsize%s_%s_%s_lr%s"%(lstmsize, dropout, n_layers, batch_size, activation, optimizer, learning_rate)

##### MAIN PART ###### 

print('Preparing data...')
start = time.time()

dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()
train, _ = dataset_manager.split_data_strict(data, train_ratio, split=data_split_type)
train, val = dataset_manager.split_val(train, val_ratio, split="random")

dt_train = dataset_manager.encode_data_with_label_all_data(train)
dt_val = dataset_manager.encode_data_with_label_all_data(val)

if "traffic_fines" in dataset_name:
    max_len = 10
elif "bpic2017" in dataset_name:
    max_len = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
else:
    max_len = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))

activity_cols = [col for col in dt_train.columns if col.startswith("act")]
n_activities = len(activity_cols)
data_dim = dt_train.shape[1] - 3

print("Done: %s"%(time.time() - start))


# compile a model with same parameters that was trained, and load the weights of the trained model
print('Training model...')
start = time.time()

main_input = Input(shape=(max_len, data_dim), name='main_input')

if n_layers == 1:
    l2_3 = LSTM(lstmsize, input_shape=(max_len, data_dim), implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=dropout)(main_input)
    b2_3 = BatchNormalization()(l2_3)
    
elif n_layers == 2:
    l1 = LSTM(lstmsize, input_shape=(max_len, data_dim), implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=dropout)(main_input) # the shared layer
    b1 = BatchNormalization(axis=1)(l1)
    l2_3 = LSTM(lstmsize, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=dropout)(b1)
    b2_3 = BatchNormalization()(l2_3)
    
elif n_layers == 3:
    l1 = LSTM(lstmsize, input_shape=(max_len, data_dim), implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=dropout)(main_input) # the shared layer
    b1 = BatchNormalization(axis=1)(l1)
    l2 = LSTM(lstmsize, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=dropout)(b1) # the shared layer
    b2 = BatchNormalization(axis=1)(l2)
    l3 = LSTM(lstmsize, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=dropout)(b2)
    b2_3 = BatchNormalization()(l3)

outcome_output = Dense(2, activation=activation, kernel_initializer='glorot_uniform', name='outcome_output')(b2_3)

model = Model(inputs=[main_input], outputs=[outcome_output])
if optimizer == "adam":
    opt = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
elif optimizer == "rmsprop":
    opt = RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss={'outcome_output':'binary_crossentropy'}, optimizer=opt)
#early_stopping = EarlyStopping(monitor='val_loss', patience=42)
#model_checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto')
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)


X, _, _, y_o = dataset_manager.generate_3d_data_with_label_all_data(dt_train, max_len)
X_val, _, _, y_o_val = dataset_manager.generate_3d_data_with_label_all_data(dt_val, max_len)

sys.stdout.flush()
history = model.fit({'main_input': X}, {'outcome_output':y_o}, validation_data=(X_val, y_o_val), verbose=2, callbacks=[lr_reducer], batch_size=batch_size, epochs=nb_epoch)

print("Done: %s"%(time.time() - start))


with open(os.path.join(output_dir, "loss_files/loss_singletask_%s_%s.csv" % (params, dataset_name)), 'w') as fout:
    fout.write("epoch;train_loss;val_loss;params\n")
    for epoch in range(len(history.history['loss'])):
        fout.write("%s;%s;%s;%s\n"%(epoch, history.history['loss'][epoch], history.history['val_loss'][epoch], params))


