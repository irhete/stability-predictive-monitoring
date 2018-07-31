"""This script trains and evaluates a predictive model for outcome-oriented predictive process monitoring.

Usage:
  python experiments_final_rf_xgboost.py <dataset> <method> <classifier> <params_dir> <results_dir> (<truncate_traces>)

Example:
  python experiments_final_rf_xgboost.py bpic2012_cancelled single_laststate xgboost_calibrated optimal_params results
  
Author: Irene Teinemaa [irene.teinemaa@gmail.com]
"""

import os
import sys
from sys import argv
import pickle
import csv

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from DatasetManager import DatasetManager
import EncoderFactory
import BucketFactory
import ClassifierFactory

from sklearn.calibration import CalibratedClassifierCV


dataset_ref = argv[1]
method_name = argv[2]
cls_method = argv[3]
params_dir = argv[4]
results_dir = argv[5]
truncate_traces = ("truncate" if len(argv) <= 6 else argv[6])

detailed_results_dir = "%s_detailed" % results_dir

bucket_method, cls_encoding = method_name.split("_")
bucket_encoding = ("last" if bucket_method == "state" else "agg")

dataset_ref_to_datasets = {
    "bpic2011": ["bpic2011_f%s"%formula for formula in range(1,5)],
    "bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(1,6)],
    "insurance": ["insurance_activity", "insurance_followup"],
    "sepsis_cases": ["sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4"]
}

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]
}

datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]
    
train_ratio = 0.8
val_ratio = 0.2
random_state = 22
min_cases_for_training = 1

# create results directories
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(detailed_results_dir):
    os.makedirs(detailed_results_dir)
    
for dataset_name in datasets:
    
    detailed_results = pd.DataFrame()
    
    # load optimal params
    optimal_params_filename = os.path.join(params_dir, "optimal_params_%s_%s_%s.pickle" % (dataset_name, method_name,
                                                                                           cls_method.replace("_calibrated", "")))
    if not os.path.isfile(optimal_params_filename) or os.path.getsize(optimal_params_filename) <= 0:
        continue
        
    with open(optimal_params_filename, "rb") as fin:
        args = pickle.load(fin)
    
    # read the data
    dataset_manager = DatasetManager(dataset_name)
    data = dataset_manager.read_dataset()

    # determine min and max (truncated) prefix lengths
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

    # split into training and test
    train, test = dataset_manager.split_data_strict(data, train_ratio)
    train, val = dataset_manager.split_val(train, val_ratio)
    overall_class_ratio = dataset_manager.get_class_ratio(train)
    
    # generate prefix logs
    dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)
    dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
            
    # Bucketing prefixes based on control flow
    bucketer_args = {'encoding_method': bucket_encoding, 
                     'case_id_col': dataset_manager.case_id_col, 
                     'cat_cols': [dataset_manager.activity_col], 
                     'num_cols': [], 
                     'random_state': random_state}
    if bucket_method == "cluster":
        bucketer_args["n_clusters"] = int(args["n_clusters"])
    cls_encoder_args = {'case_id_col': dataset_manager.case_id_col, 
                        'static_cat_cols': dataset_manager.static_cat_cols,
                        'static_num_cols': dataset_manager.static_num_cols, 
                        'dynamic_cat_cols': dataset_manager.dynamic_cat_cols,
                        'dynamic_num_cols': dataset_manager.dynamic_num_cols, 
                        'fillna': True}
    bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
    bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
    bucket_assignments_test = bucketer.predict(dt_test_prefixes)
    
    if "calibrate" in cls_method:
        dt_val_prefixes = dataset_manager.generate_prefix_data(val, min_prefix_length, max_prefix_length)
        bucket_assignments_val = bucketer.predict(dt_val_prefixes)


    preds_all = []
    test_y_all = []
    nr_events_all = []
    for bucket in set(bucket_assignments_test):
        current_args = args if bucket_method != "prefix" else args[bucket]
        current_args["n_estimators"] = 500
            
        # select prefixes for the given bucket
        relevant_train_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
        relevant_test_cases_bucket = dataset_manager.get_indexes(dt_test_prefixes)[bucket_assignments_test == bucket]
        
        dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_train_cases_bucket)
        dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_test_cases_bucket)

        train_y = dataset_manager.get_label_numeric(dt_train_bucket)
        test_y = dataset_manager.get_label_numeric(dt_test_bucket)
            
        # add data about prefixes in this bucket (class labels and prefix lengths)
        nr_events_all.extend(list(dataset_manager.get_prefix_lengths(dt_test_bucket)))
        test_y_all.extend(test_y)

        # encode the prefixes
        feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
        if "svm" in cls_method or "logit" in cls_method:
            feature_combiner = Pipeline([('encoder', feature_combiner), ('scaler', StandardScaler())])
            
        X_train = feature_combiner.fit_transform(dt_train_bucket)
        X_test = feature_combiner.transform(dt_test_bucket)

        # fit classifier and calibrate
        cls = ClassifierFactory.get_classifier(cls_method.replace("_calibrated", ""), current_args, random_state, min_cases_for_training, overall_class_ratio, binary=(False if "calibrate" in cls_method else True))
        cls.fit(X_train, train_y)

        if "calibrate" in cls_method:
            relevant_val_cases_bucket = dataset_manager.get_indexes(dt_val_prefixes)[bucket_assignments_val == bucket]
            dt_val_bucket = dataset_manager.get_relevant_data_by_indexes(dt_val_prefixes, relevant_val_cases_bucket)

            X_val = feature_combiner.transform(dt_val_bucket)
            y_val = dataset_manager.get_label_numeric(dt_val_bucket)
            
            cls = CalibratedClassifierCV(cls, cv="prefit", method='sigmoid')
            cls.fit(X_val, np.array(y_val))

        # predict 
        preds = cls.predict_proba(X_test)
        if "calibrate" in cls_method:
            preds = preds[:,1]
        preds_all.extend(preds)
        
        case_ids = list(dt_test_bucket.groupby(dataset_manager.case_id_col).first().index)
        current_results = pd.DataFrame({"dataset": dataset_name, "method": method_name, "cls": cls_method, 
                                    "nr_events": bucket, "predicted": preds, "actual": test_y, "case_id": case_ids})
        detailed_results = pd.concat([detailed_results, current_results], axis=0)

    # write results
    outfile = os.path.join(results_dir, "results_%s_%s_%s.csv" % (cls_method, dataset_name, method_name))
    detailed_results_file = os.path.join(detailed_results_dir, "detailed_results_%s_%s_%s.csv"%(cls_method, dataset_name, method_name)) 
    detailed_results.to_csv(detailed_results_file, sep=";", index=False)

    with open(outfile, 'w') as csvfile:
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
