"""This script extracts the best parameters for a predictive model based on multiple training runs. Execute this script after testing several parameter settings using the script experiments_param_optim_lstm.py.

Usage:
  python extract_best_params_lstm.py

Author: Irene Teinemaa [irene.teinemaa@gmail.com]
"""

import glob
import os
import pickle
from sys import argv

import numpy as np
import pandas as pd

loss_files_dir = "val_results_lstm"
params_dir = "optimal_params_lstm"

if not os.path.exists(params_dir):
    os.makedirs(params_dir)

datasets = ["bpic2012_accepted", "bpic2012_cancelled", "bpic2012_declined", "bpic2017_accepted", "bpic2017_cancelled", 
            "bpic2017_refused", "sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4", "production", "traffic_fines_1",
            "hospital_billing_3"]
method_names = ["lstm"]
cls_methods = ["lstm"]

cls_params_names = ['lstmsize', 'dropout', 'n_layers', 'batch_size', 'optimizer', 'learning_rate', 'nb_epoch']

for dataset_name in datasets:
    for method_name in method_names:
        for cls_method in cls_methods:
            files = glob.glob("%s/%s" % (loss_files_dir, "loss_%s_%s_*.csv" % (dataset_name, method_name)))
            if len(files) < 1:
                continue
            dt_all = pd.DataFrame()
            for file in files:
                dt_all = pd.concat([dt_all, pd.read_csv(file, sep=";")], axis=0)

            dt_all = dt_all[dt_all["epoch"] >= 5]
            dt_all["params"] = dt_all["params"] + "_" + dt_all["epoch"].astype(str)

            cls_params_str = dt_all["params"][np.argmin(dt_all["val_loss"])]

            best_params = {cls_params_names[i]: val for i, val in enumerate(cls_params_str.split("_"))}
            outfile = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (dataset_name, method_name,
                                                                                                   cls_method))
            with open(outfile, "wb") as fout:
                pickle.dump(best_params, fout)

