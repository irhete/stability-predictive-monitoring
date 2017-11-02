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
from sklearn.metrics import mean_absolute_error, accuracy_score, roc_auc_score
import os
import sys
from dataset_manager import DatasetManager
from sys import argv
import glob

import pandas as pd
import numpy as np


dataset_name = argv[1]
cls_method = "lstm"

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
params = "lstm_final"
checkpoint_prefix = os.path.join(output_dir, "checkpoints/model_%s_%s"%(dataset_name, params))
checkpoint_filepath = "%s.{epoch:02d}-{val_loss:.2f}.hdf5"%checkpoint_prefix
#params = "lstmsize%s_dropout%s_nlayers%s_batchsize%s_%s_%s_lr%s"%(lstmsize, dropout, n_layers, batch_size, activation, optimizer, learning_rate)
model_filename = glob.glob("%s*.hdf5"%checkpoint_prefix)[-1]
results_file = os.path.join(output_dir, "evaluation_results/results_%s_%s_%s.csv"%(cls_method, dataset_name, params))
detailed_results_file = os.path.join(output_dir, "evaluation_results_detailed/results_%s_%s_%s.csv"%(cls_method, dataset_name, params))

##### MAIN PART ###### 

print('Preparing data...')
start = time.time()

dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()
train, test = dataset_manager.split_data_strict(data, train_ratio, split=data_split_type)
train, val = dataset_manager.split_val(train, val_ratio, split="random")

dt_train = dataset_manager.encode_data_with_label_all_data(train)
dt_val = dataset_manager.encode_data_with_label_all_data(val)
dt_test = dataset_manager.encode_data_with_label_all_data(test)
print(dt_train.shape, dt_val.shape, dt_test.shape)


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
model.load_weights(model_filename)

print('Evaluating...')
start = time.time()
detailed_results = pd.DataFrame()
with open(results_file, 'w') as fout:
    csv_writer = csv.writer(fout, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["dataset", "cls", "params", "nr_events", "metric", "score"])
    
    total = 0
    total_auc_outcome = 0
    for nr_events in range(1, max_len+1):
        # encode only prefixes of this length
        X, y_a, y_t, y_o, case_ids = dataset_manager.generate_3d_data_for_prefix_length_with_label_all_data(dt_test, max_len, nr_events)
        print(X.shape, y_a.shape, y_t.shape, y_o.shape)
        if X.shape[0] == 0:
            break
        
        #y_t = y_t * dataset_manager.divisors["timesincelastevent"]
        
        pred_y_o = model.predict(X, verbose=0)
        try:
            auc_outcome = roc_auc_score(y_o[:,0], pred_y_o[:,0])
        except ValueError:
            auc_outcome = 0.5
        total += X.shape[0]
        total_auc_outcome += auc_outcome * X.shape[0]
        
        print("prefix = %s, n_cases = %s, auc_outcome = %s"%(nr_events, X.shape[0], auc_outcome))
        csv_writer.writerow([dataset_name, cls_method, params, nr_events, "n_cases", X.shape[0]])
        csv_writer.writerow([dataset_name, cls_method, params, nr_events, "auc_outcome", auc_outcome])

        current_results = pd.DataFrame({"dataset": dataset_name, "cls": cls_method, "params": params, "nr_events": nr_events, "predicted": pred_y_o[:,0], "actual": y_o[:,0], "case_id": case_ids})
        detailed_results = pd.concat([detailed_results, current_results], axis=0)
        
    csv_writer.writerow([dataset_name, cls_method, params, -1, "total_auc_outcome", total_auc_outcome / total])
    
print("Done: %s"%(time.time() - start))
        
print("total auc_outcome: ", total_auc_outcome / total)

detailed_results.to_csv(detailed_results_file, sep=";", index=False)
