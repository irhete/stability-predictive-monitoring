"""This script trains and evaluates and a deep neural network (with LSTM units) based predictive model for outcome-oriented predictive process monitoring. The script writes the predictions for a test set to a file.

***
The architecture of the neural network is based on the approach proposed in the following paper:
Niek Tax, Ilya Verenich, Marcello La Rosa, Marlon Dumas: 
Predictive Business Process Monitoring with LSTM Neural Networks. CAiSE 2017: 477-492,
with code available at: https://github.com/verenich/ProcessSequencePrediction
***

Usage of the current script:
  python experiments_final_lstm.py <dataset> <method> <classifier> <params_dir> <results_dir>

Example:
  python experiments_final_lstm.py bpic2012_cancelled lstm lstm_calibrated optimal_params_lstm results_lstm
  
Author: Irene Teinemaa [irene.teinemaa@gmail.com]
"""

import time
import os
from sys import argv
import csv
import pickle

import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.layers import Input
from keras.optimizers import Nadam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

from DatasetManager import DatasetManager
import auc_callback
from calibration_models import LSTM2D


dataset_name = argv[1]
method_name = argv[2]
cls_method = argv[3]
params_dir = argv[4]
results_dir = argv[5]

optimal_params_filename = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (dataset_name, method_name, 
                                                                                       cls_method.replace("_calibrated", "")))
with open(optimal_params_filename, "rb") as fin:
    params = pickle.load(fin)
    
lstmsize = int(params['lstmsize'])
dropout = float(params['dropout'])
n_layers = int(params['n_layers'])
batch_size = int(params['batch_size'])
optimizer = params['optimizer']
learning_rate = float(params['learning_rate'])
nb_epoch = int(params['nb_epoch'])

activation = "sigmoid"
train_ratio = 0.8
val_ratio = 0.2

detailed_results_dir = "%s_detailed" % results_dir
# create results directories
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(detailed_results_dir):
    os.makedirs(detailed_results_dir)

##### MAIN PART ###### 

print('Preparing data...')
start = time.time()

dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()
train, test = dataset_manager.split_data_strict(data, train_ratio)
train, val = dataset_manager.split_val(train, val_ratio)

if "traffic_fines" in dataset_name:
    max_len = 10
elif "bpic2017" in dataset_name:
    max_len = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
else:
    max_len = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))
del data
    
dt_train = dataset_manager.encode_data_for_lstm(train)
del train
data_dim = dt_train.shape[1] - 3
X, y = dataset_manager.generate_3d_data(dt_train, max_len)
del dt_train

dt_val = dataset_manager.encode_data_for_lstm(val)
del val
X_val, y_val = dataset_manager.generate_3d_data(dt_val, max_len)
del dt_val

dt_test = dataset_manager.encode_data_for_lstm(test)
del test

print("Done: %s"%(time.time() - start))


# compile a model with same parameters that was trained, and load the weights of the trained model
print('Training model...')
start = time.time()

main_input = Input(shape=(max_len, data_dim), name='main_input')

if n_layers == 1:
    l2_3 = LSTM(lstmsize, input_shape=(max_len, data_dim), implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=dropout)(main_input)
    b2_3 = BatchNormalization()(l2_3)
    
elif n_layers == 2:
    l1 = LSTM(lstmsize, input_shape=(max_len, data_dim), implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=dropout)(main_input)
    b1 = BatchNormalization(axis=1)(l1)
    l2_3 = LSTM(lstmsize, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=dropout)(b1)
    b2_3 = BatchNormalization()(l2_3)
    
elif n_layers == 3:
    l1 = LSTM(lstmsize, input_shape=(max_len, data_dim), implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=dropout)(main_input)
    b1 = BatchNormalization(axis=1)(l1)
    l2 = LSTM(lstmsize, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=dropout)(b1)
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

auc_cb = auc_callback.AUCHistory(X_val, y_val)
history = model.fit({'main_input': X}, {'outcome_output':y}, validation_data=(X_val, y_val), verbose=2, callbacks=[auc_cb], batch_size=batch_size, epochs=nb_epoch)

if "calibrate" in cls_method:
    n_cases, time_dim, n_features = X_val.shape
    model_2d = LSTM2D(model, time_dim, n_features)
    model_calibrated = CalibratedClassifierCV(model_2d, cv="prefit", method='sigmoid')
    model_calibrated.fit(X_val.reshape(n_cases, time_dim*n_features), y_val[:,1])

print("Done: %s"%(time.time() - start))


# Write loss for each epoch
print('Evaluating...')
start = time.time()

detailed_results = pd.DataFrame()
preds_all = []
test_y_all = []
nr_events_all = []
for nr_events in range(1, max_len+1):
    # encode only prefixes of this length
    X, y, case_ids = dataset_manager.generate_3d_data_for_prefix_length(dt_test, max_len, nr_events)

    if X.shape[0] == 0:
        break

    if "calibrate" in cls_method:
        preds = model_calibrated.predict_proba(X.reshape(X.shape[0], time_dim*n_features))[:,1]
    else:
        preds = model.predict(X, verbose=0)[:,1]

    current_results = pd.DataFrame({"dataset": dataset_name, "method": method_name, "cls": cls_method,
                                    "nr_events": nr_events, "predicted": preds, "actual": y[:,1], "case_id": case_ids})
    detailed_results = pd.concat([detailed_results, current_results], axis=0)
    
    preds_all.extend(preds)
    test_y_all.extend(y[:,1])
    nr_events_all.extend([nr_events] * X.shape[0])
    
print("Done: %s"%(time.time() - start))
        
# Write results
results_file = os.path.join(results_dir, "results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
detailed_results_file = os.path.join(detailed_results_dir, "detailed_results_%s_%s_%s.csv"%(cls_method, dataset_name, method_name)) 
detailed_results.to_csv(detailed_results_file, sep=";", index=False)

with open(results_file, 'w') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)
    spamwriter.writerow(["dataset", "method", "cls", "nr_events", "metric", "score"])

    dt_results = pd.DataFrame({"actual": test_y_all, "predicted": preds_all, "nr_events": nr_events_all})
    for nr_events, group in dt_results.groupby("nr_events"):
        auc = np.nan if len(set(group.actual)) < 2 else roc_auc_score(group.actual, group.predicted)
        spamwriter.writerow([dataset_name, method_name, cls_method, nr_events, -1, "auc", auc])
        print(nr_events, auc)

    auc = roc_auc_score(dt_results.actual, dt_results.predicted)
    spamwriter.writerow([dataset_name, method_name, cls_method, -1, "auc", auc])
    print(auc)