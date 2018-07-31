"""This script extracts the best parameters for a predictive model based on a single training run. Execute this script after testing several parameter settings using the script experiments_param_optim_rf_xgboost.py.

Usage:
  python extract_best_params_single_run_rf_xgboost.py

Author: Irene Teinemaa [irene.teinemaa@gmail.com]
"""

import glob
import os
import pickle
import operator
from sys import argv

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


preds_dir = "val_results"
params_dir = "optimal_params"

if not os.path.exists(params_dir):
    os.makedirs(params_dir)

datasets = ["bpic2012_accepted", "bpic2012_cancelled", "bpic2012_declined", "bpic2017_accepted", "bpic2017_cancelled", 
            "bpic2017_refused", "sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4", "production", "traffic_fines_1",
            "hospital_billing_3"]
method_names = ["single_agg", "prefix_index", "single_index"]
cls_methods = ["rf", "xgboost"]

cls_params_names = {'rf': ['n_estimators', 'max_features'],
                    'xgboost': ['n_estimators', 'learning_rate', 'subsample', 'max_depth', 'colsample_bytree', 
                                'min_child_weight']}

for dataset_name in datasets:
    for method_name in method_names:
        for cls_method in cls_methods:
            files = glob.glob("%s/%s" % (preds_dir, "val_results_%s_%s_%s_*.csv"%(dataset_name, method_name, cls_method)))
            if len(files) < 1:
                continue
            metrics = {}
            for file in files:
                data = pd.read_csv(file, sep=";")
                
                # calculate auc for given parameters
                if "index" in method_name:
                    for nr_events, group in data.groupby("nr_events"):
                        if nr_events not in metrics:
                            metrics[nr_events] = {}
                        metrics[nr_events][data["params"].iloc[0]] = roc_auc_score(group.actual, group.predicted)
                else:
                    metrics[data["params"].iloc[0]] = roc_auc_score(data.actual, data.predicted)

            if "index" in method_name:
                best_params = {}
                for nr_events, vals in metrics.items():
                    cls_params_str = max(vals.items(), key=operator.itemgetter(1))[0]
                    best_params[nr_events] = {cls_params_names[cls_method][i]: val for i, val in enumerate(cls_params_str.split("_"))}
            else:
                cls_params_str = max(metrics.items(), key=operator.itemgetter(1))[0]
                best_params = {cls_params_names[cls_method][i]: val for i, val in enumerate(cls_params_str.split("_"))}
                
            outfile = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (dataset_name, method_name,
                                                                                                   cls_method))
            with open(outfile, "wb") as fout:
                pickle.dump(best_params, fout)
                