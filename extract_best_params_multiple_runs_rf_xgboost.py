"""This script extracts the best parameters for a predictive model based on multiple training runs. Execute this script after testing several parameter settings using the script experiments_param_optim_rf_xgboost.py.

Usage:
  python extract_best_params_multiple_runs_rf_xgboost.py

Author: Irene Teinemaa [irene.teinemaa@gmail.com]
"""

import glob
import os
import pickle
from sys import argv

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


preds_dir = "val_results_runs"
params_dir_auc = "optimal_params_5runs_auc"
params_dir_auc_stab = "optimal_params_5runs_auc_rmspd"

if not os.path.exists(params_dir_auc):
    os.makedirs(params_dir_auc)
if not os.path.exists(params_dir_auc_stab):
    os.makedirs(params_dir_auc_stab)

datasets = ["bpic2012_accepted", "bpic2012_cancelled", "bpic2012_declined", "bpic2017_accepted", "bpic2017_cancelled", 
            "bpic2017_refused", "sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4", "production", "traffic_fines_1",
            "hospital_billing_3"]
method_names = ["single_agg"]
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
                grouped = data.groupby("run")

                aucs = []
                for gr, group in grouped:
                    aucs.append(roc_auc_score(group.actual, group.predicted))

                mspd_acc = 0
                n_runs = np.max(data.run) + 1
                for i in range(n_runs):
                    tmp1 = data[data.run==i]
                    for j in range(i):
                        tmp2 = data[data.run==j]
                        tmp_merged = tmp1.merge(tmp2, on=["case_id", "dataset_name", "nr_events", "params"])

                        mspd_acc += 2.0 / (n_runs * (n_runs - 1)) * np.mean(np.power(tmp_merged.predicted_x - tmp_merged.predicted_y, 2))

                metrics[data["params"].iloc[0]] = {}
                metrics[data["params"].iloc[0]]["auc"] = np.mean(aucs)
                metrics[data["params"].iloc[0]]["rmspd"] = np.sqrt(mspd_acc)

            dt_metrics = pd.DataFrame.from_dict(metrics, orient="index")

            # choose best params according to AUC only
            alpha = 1
            beta = 0
            score = alpha * dt_metrics.auc - beta * dt_metrics.rmspd
            cls_params_str = np.argmax(score)
            best_params = {cls_params_names[cls_method][i]: val for i, val in enumerate(cls_params_str.split("_"))}
            outfile = os.path.join(params_dir_auc, "optimal_params_%s_%s_%s.pickle" % (dataset_name, method_name,
                                                                                                   cls_method))
            with open(outfile, "wb") as fout:
                pickle.dump(best_params, fout)
                
            # choose best params according to AUC and RMSPDE
            alpha = 1
            beta = 5
            score = alpha * dt_metrics.auc - beta * dt_metrics.rmspd
            cls_params_str = np.argmax(score)
            best_params = {cls_params_names[cls_method][i]: val for i, val in enumerate(cls_params_str.split("_"))}
            outfile = os.path.join(params_dir_auc_stab, "optimal_params_%s_%s_%s.pickle" % (dataset_name, 
                                                                                                         method_name,
                                                                                                         cls_method))
            with open(outfile, "wb") as fout:
                pickle.dump(best_params, fout)

