import numpy as np
import scipy.sparse

import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, FeatureUnion
from time import time
import pickle
import os
from sys import argv
import itertools

import EncoderFactory
import ClassifierFactory
import BucketFactory
from DatasetManager import DatasetManager

dataset_ref = argv[1]
bucket_encoding = argv[2]
bucket_method = argv[3]
cls_encoding = argv[4]
cls_method = argv[5]
results_dir = argv[6]
n_iter = int(argv[7])

method_name = "%s_%s"%(bucket_method, cls_encoding)

home_dir = ""

if not os.path.exists(os.path.join(home_dir, results_dir)):
    os.makedirs(os.path.join(home_dir, results_dir))

dataset_ref_to_datasets = {
    "bpic2011": ["bpic2011_f%s"%formula for formula in range(1,5)],
    "bpic2011_orig_order": ["bpic2011_f%s_orig_order"%formula for formula in range(1,5)],
    "bpic2015": ["bpic2015_%s_f2"%(municipality) for municipality in range(1,6)],
    "insurance": ["insurance_activity", "insurance_followup"],
    "bpic2017": ["bpic2017"],
    "sepsis_cases": ["sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4"]
}

encoding_dict = {
    "laststate": ["static", "last"],
    "agg": ["static", "agg"],
    "index": ["static", "index"],
    "combined": ["static", "last", "agg"]}
    
datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]

outfile = os.path.join(home_dir, results_dir, "val_results_%s_%s_%s.csv"%(cls_method, method_name, dataset_ref)) 

train_ratio = 0.8
val_ratio = 0.2
random_state = 22
fillna = True
n_min_cases_in_bucket = 30

cls_params_names = ['n_estimators', 'max_features']

def loguniform(low=0, high=1):
    val = np.exp(np.random.uniform(0, 1, None))
    scaled_val = (((val - np.exp(0)) * (high - low)) / (np.exp(1) - np.exp(0))) + low
    return scaled_val

##### MAIN PART ######    
with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s;%s;%s;%s;%s\n"%("part", "dataset", "method", "cls", ";".join(cls_params_names), "nr_events", "metric", "score"))
    
    for dataset_name in datasets:
        
        dataset_manager = DatasetManager(dataset_name)
        data = dataset_manager.read_dataset()
        train, _ = dataset_manager.split_data_strict(data, train_ratio)
        train, test = dataset_manager.split_val(train, val_ratio)
         # consider prefix lengths until 90% of positive cases have finished
        min_prefix_length = 1
        if "traffic_fines" in dataset_name:
            max_prefix_length = 10
        elif "bpic2017" in dataset_name:
            max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
        else:
            max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))
        del data
        min_prefix_length = 1
        part = 0
        
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


        dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
        dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)

        # Bucketing prefixes based on control flow
        print("Bucketing prefixes...")
        bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
        bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)

        for i in range(n_iter):
            n_estimators = np.random.randint(150, 1000)
            max_features = loguniform(0.01, 0.9)

            params = {'n_estimators': n_estimators,
                      'max_features': max_features}

            pipelines = {}

            # train and fit pipeline for each bucket
            for bucket in set(bucket_assignments_train):
                print("Fitting pipeline for bucket %s..."%bucket)
                relevant_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
                dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_cases_bucket) # one row per event
                train_y = dataset_manager.get_label_numeric(dt_train_bucket)

                feature_combiner = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
                pipelines[bucket] = Pipeline([('encoder', feature_combiner), ('cls', ClassifierFactory.get_classifier(cls_method, **params))])
                pipelines[bucket].fit(dt_train_bucket, train_y)

            # if the bucketing is prefix-length-based, then evaluate for each prefix length separately, otherwise evaluate all prefixes together 
            max_evaluation_prefix_length = max_prefix_length if bucket_method == "prefix" else min_prefix_length

            prefix_lengths_test = dt_test_prefixes.groupby(dataset_manager.case_id_col).size()
            
            for nr_events in range(min_prefix_length, max_evaluation_prefix_length+1):
                print("Predicting for %s events..."%nr_events)

                if bucket_method == "prefix":
                    # select only prefixes that are of length nr_events
                    relevant_cases_nr_events = prefix_lengths_test[prefix_lengths_test == nr_events].index

                    if len(relevant_cases_nr_events) == 0:
                        break

                    dt_test_nr_events = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_cases_nr_events)
                    del relevant_cases_nr_events
                else:
                    # evaluate on all prefixes
                    dt_test_nr_events = dt_test_prefixes.copy()

                start = time()
                # get predicted cluster for each test case
                bucket_assignments_test = bucketer.predict(dt_test_nr_events)

                # use appropriate classifier for each bucket of test cases
                # for evaluation, collect predictions from different buckets together
                preds = []
                test_y = []
                for bucket in set(bucket_assignments_test):
                    relevant_cases_bucket = dataset_manager.get_indexes(dt_test_nr_events)[bucket_assignments_test == bucket]
                    dt_test_bucket = dataset_manager.get_relevant_data_by_indexes(dt_test_nr_events, relevant_cases_bucket) # one row per event

                    if len(relevant_cases_bucket) == 0:
                        continue

                    elif bucket not in pipelines:
                        # use the general class ratio (in training set) as prediction 
                        preds_bucket = [dataset_manager.get_class_ratio(train_chunk)] * len(relevant_cases_bucket)

                    else:
                        # make actual predictions
                        preds_bucket = pipelines[bucket].predict_proba(dt_test_bucket)

                    preds.extend(preds_bucket)

                    # extract actual label values
                    test_y_bucket = dataset_manager.get_label_numeric(dt_test_bucket) # one row per case
                    test_y.extend(test_y_bucket)

                if len(set(test_y)) < 2:
                    auc = None
                else:
                    auc = roc_auc_score(test_y, preds)
                cls_params_str = ";".join([str(params[param]) for param in cls_params_names])

                fout.write("%s;%s;%s;%s;%s;%s;%s;%s\n"%(part, dataset_name, method_name, cls_method, cls_params_str, nr_events, "auc", auc))
            