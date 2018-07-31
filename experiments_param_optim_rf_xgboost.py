"""This script trains a predictive model over a single or multiple runs (i.e. same model configuration but different random seeds) and writes the predictions for each run to a file. The predictive models are intended for outcome-oriented predictive process monitoring.

Usage:
  experiments_param_optim_runs.py <dataset> <method> <classifier> <params_str> <n_runs> <preds_dir> (<truncate_traces>)

Example:
  experiments_param_optim_runs.py bpic2012_cancelled single_laststate xgboost 100_0.2_3_5_0.1_10 5 val_results
  
Author: Irene Teinemaa [irene.teinemaa@gmail.com]
"""

import os
from sys import argv

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion

import EncoderFactory
import ClassifierFactory
import BucketFactory
from DatasetManager import DatasetManager

dataset_name = argv[1]
method_name = argv[2]
cls_method = argv[3]
params_str = argv[4]
n_runs = int(argv[5]) # 5
results_dir = argv[6]
truncate_traces = ("truncate" if len(argv) <= 7 else argv[7])

bucket_method, cls_encoding = method_name.split("_")
bucket_encoding = ("last" if bucket_method == "state" else "agg")


cls_params_names = {'rf': ['n_estimators', 'max_features'],
                    'xgboost': ['n_estimators', 'learning_rate', 'subsample', 'max_depth', 'colsample_bytree', 
                                'min_child_weight']}

params = {cls_params_names[cls_method][i]: val for i, val in enumerate(params_str.split("_"))}
# convert to int
for param in ['n_estimators', 'max_depth', 'min_child_weight']:
    if param in params:
        params[param] = int(params[param])
# convert to float
for param in ['max_features', 'learning_rate', 'subsample', 'colsample_bytree']:
    if param in params:
        params[param] = float(params[param])

cls_params_str = "_".join([str(params[param]) for param in cls_params_names[cls_method]])

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]}
    
methods = encoding_dict[cls_encoding]

outfile = os.path.join(results_dir, "val_results_%s_%s_%s_%s.csv"%(dataset_name, method_name, cls_method, cls_params_str)) 

train_ratio = 0.8
val_ratio = 0.2
random_state = 22
fillna = True
min_cases_for_training = 1

##### MAIN PART ######    
dataset_manager = DatasetManager(dataset_name)
data = dataset_manager.read_dataset()
train, _ = dataset_manager.split_data_strict(data, train_ratio)
train, test = dataset_manager.split_val(train, val_ratio)

# consider prefix lengths until 90% of positive cases have finished
min_prefix_length = 1
if truncate_traces == "truncate":
    if "traffic_fines" in dataset_name:
        max_prefix_length = 10
    elif "bpic2017" in dataset_name:
        max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
    else:
        max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))
else:
    max_prefix_length = dataset_manager.get_max_case_length(data)
del data

# extract arguments
bucketer_args = {'encoding_method':bucket_encoding, 
                 'case_id_col':dataset_manager.case_id_col, 
                 'cat_cols':[dataset_manager.activity_col], 
                 'num_cols':[], 
                 'random_state':random_state}

cls_encoder_args = {'case_id_col':dataset_manager.case_id_col, 
                    'static_cat_cols':dataset_manager.static_cat_cols,
                    'static_num_cols':dataset_manager.static_num_cols, 
                    'dynamic_cat_cols':dataset_manager.dynamic_cat_cols,
                    'dynamic_num_cols':dataset_manager.dynamic_num_cols, 
                    'fillna':fillna}

overall_class_ratio = dataset_manager.get_class_ratio(train)
dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)

# Bucketing prefixes based on control flow
print("Bucketing prefixes...")
bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
bucket_assignments_test = bucketer.predict(dt_test_prefixes)

# iterate for number of runs
dt_all_predictions = pd.DataFrame()
for current_run in range(n_runs):
    for bucket in set(bucket_assignments_test):
        # select prefixes for the given bucket
        relevant_train_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
        relevant_test_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[bucket_assignments_test == bucket]
        dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_test_cases_bucket)
        dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_train_cases_bucket)
        train_y = dataset_manager.get_label_numeric(dt_train_bucket)
        
        # extract data about prefixes in this bucket (class labels and prefix lengths)
        test_y = dataset_manager.get_label_numeric(dt_test_bucket)
        test_nr_events = list(dataset_manager.get_prefix_lengths(dt_test_bucket))
        test_case_ids = list(dt_test_bucket.groupby(dataset_manager.case_id_col).first().index)
            
        # initialize pipeline for sequence encoder and classifier
        feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
        cls = ClassifierFactory.get_classifier(cls_method, params, None, min_cases_for_training, overall_class_ratio)
        pipeline = Pipeline([('encoder', feature_combiner), ('cls', cls)])

        # fit pipeline
        pipeline.fit(dt_train_bucket, train_y)

        # predict 
        preds = pipeline.predict_proba(dt_test_bucket)

        dt_all_predictions = pd.concat([dt_all_predictions, pd.DataFrame({"predicted": preds,
                                                                          "actual": test_y,
                                                                          "case_id": test_case_ids,
                                                                          "dataset_name": dataset_name,
                                                                          "nr_events": test_nr_events,
                                                                          "run": current_run,
                                                                          "params": cls_params_str,
                                                                          "method": method_name,
                                                                          "cls": cls_method})], axis=0)
dt_all_predictions.to_csv(outfile, sep=";", index=False)
