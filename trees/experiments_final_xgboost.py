import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.calibration import CalibratedClassifierCV
from time import time
import pickle
import os
from sys import argv
import xgboost as xgb

import EncoderFactory
import BucketFactory
import ClassifierFactory
from DatasetManager import DatasetManager

dataset_ref = argv[1]
bucket_encoding = argv[2]
bucket_method = argv[3]
cls_encoding = argv[4]
cls_method = argv[5]
optimal_params_filename = argv[6]
results_dir = argv[7]

cls_method2 = cls_method
if "calibrated" in cls_method:
    cls_method = cls_method.split("_")[0]

detailed_results_dir = "%s_detailed" % results_dir

method_name = "%s_%s"%(bucket_method, cls_encoding)

home_dir = ""

if not os.path.exists(os.path.join(home_dir, results_dir)):
    os.makedirs(os.path.join(home_dir, results_dir))

with open(os.path.join(home_dir, optimal_params_filename), "rb") as fin:
    best_params = pickle.load(fin)

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
    "skipgram": ["static", "index", "skipgram"],
    "combined": ["static", "last", "agg"]}
    
datasets = [dataset_ref] if dataset_ref not in dataset_ref_to_datasets else dataset_ref_to_datasets[dataset_ref]
methods = encoding_dict[cls_encoding]

outfile = os.path.join(home_dir, results_dir, "final_results_%s_%s_%s.csv"%(cls_method2, method_name, dataset_ref)) 
detailed_results_file = os.path.join(home_dir, detailed_results_dir, "detailed_results_%s_%s_%s.csv"%(cls_method2, method_name, dataset_ref)) 
    
train_ratio = 0.8
val_ratio = 0.2
random_state = 22
fillna = True
n_min_cases_in_bucket = 30
    
    
##### MAIN PART ######    
detailed_results = pd.DataFrame()
with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s;%s;%s\n"%("dataset", "method", "cls", "nr_events", "metric", "score"))
    
    for dataset_name in datasets:
        
        dataset_manager = DatasetManager(dataset_name)
        
        # read the data
        data = dataset_manager.read_dataset()
        
        # split data into train and test
        train, test = dataset_manager.split_data_strict(data, train_ratio)
        train, val = dataset_manager.split_val(train, val_ratio)
        
        # consider prefix lengths until 90% of positive cases have finished
        min_prefix_length = 1
        if "traffic_fines" in dataset_name:
            max_prefix_length = 10
        elif "bpic2017" in dataset_name:
            max_prefix_length = min(20, dataset_manager.get_pos_case_length_quantile(data, 0.90))
        else:
            max_prefix_length = min(40, dataset_manager.get_pos_case_length_quantile(data, 0.90))
        del data

        # create prefix logs
        dt_train_prefixes = dataset_manager.generate_prefix_data(train, min_prefix_length, max_prefix_length)
        dt_val_prefixes = dataset_manager.generate_prefix_data(val, min_prefix_length, max_prefix_length)
        dt_test_prefixes = dataset_manager.generate_prefix_data(test, min_prefix_length, max_prefix_length)

        print(dt_train_prefixes.shape)
        print(dt_val_prefixes.shape)
        print(dt_test_prefixes.shape)
        
        # extract arguments
        bucketer_args = {'encoding_method':bucket_encoding, 
                         'case_id_col':dataset_manager.case_id_col, 
                         'cat_cols':[dataset_manager.activity_col], 
                         'num_cols':[], 
                         'n_clusters':None, 
                         'random_state':random_state}
        if bucket_method == "cluster":
            if dataset_name not in best_params or method_name not in best_params[dataset_name] or cls_method not in best_params[dataset_name][method_name] or 'n_clusters' not in best_params[dataset_name][method_name][cls_method]:
                print("Using default value for n_clusters")
                bucketer_args['n_clusters'] = 5
            else:
                bucketer_args['n_clusters'] = best_params[dataset_name][method_name][cls_method]['n_clusters']
        
        cls_encoder_args = {'case_id_col':dataset_manager.case_id_col, 
                            'static_cat_cols':dataset_manager.static_cat_cols,
                            'static_num_cols':dataset_manager.static_num_cols, 
                            'dynamic_cat_cols':dataset_manager.dynamic_cat_cols,
                            'dynamic_num_cols':dataset_manager.dynamic_num_cols, 
                            #'activity_col':dataset_manager.activity_col, 
                            #'timestamp_col':dataset_manager.timestamp_col, 
                            'fillna':fillna}
        
        
        # Bucketing prefixes based on control flow
        print("Bucketing prefixes...")
        bucketer = BucketFactory.get_bucketer(bucket_method, **bucketer_args)
        bucket_assignments_train = bucketer.fit_predict(dt_train_prefixes)
        bucket_assignments_val = bucketer.fit_predict(dt_val_prefixes)
        bucket_assignments_test = bucketer.fit_predict(dt_test_prefixes)
            
        feature_combiners = {}
        classifiers = {}

        # train and fit pipeline for each bucket
        for bucket in set(bucket_assignments_train):
            print("Fitting pipeline for bucket %s..."%bucket)
            
            # set optimal params for this bucket
            if bucket_method == "prefix":
                cls_args = best_params[dataset_name][method_name][cls_method][bucket]
                
            else:
                cls_args = best_params[dataset_name][method_name][cls_method]
            
            cls_args["n_estimators"] = 1000
            #cls_args['random_state'] = random_state
            #cls_args['min_cases_for_training'] = n_min_cases_in_bucket
        
            # select relevant cases
            relevant_cases_bucket = dataset_manager.get_indexes(dt_train_prefixes)[bucket_assignments_train == bucket]
            relevant_cases_bucket_val = dataset_manager.get_indexes(dt_val_prefixes)[bucket_assignments_val == bucket]
            dt_train_bucket = dataset_manager.get_relevant_data_by_indexes(dt_train_prefixes, relevant_cases_bucket) # one row per event
            dt_val_bucket = dataset_manager.get_relevant_data_by_indexes(dt_val_prefixes, relevant_cases_bucket_val) # one row per event
            y_train = dataset_manager.get_label_numeric(dt_train_bucket)
            y_val = dataset_manager.get_label_numeric(dt_val_bucket)
            if len(set(y_train)) < 2 or len(set(y_val)) < 2:
                break
            
            feature_combiners[bucket] = FeatureUnion([(method, EncoderFactory.get_encoder(method, **cls_encoder_args)) for method in methods])
            
            classifiers[bucket] = xgb.XGBClassifier(objective='binary:logistic', **cls_args)
            
            X_train = feature_combiners[bucket].fit_transform(dt_train_bucket)
            X_val = feature_combiners[bucket].fit_transform(dt_val_bucket)

            eval_set = [(X_val, y_val)]
            classifiers[bucket].fit(X_train, y_train, early_stopping_rounds=10, eval_metric="auc", eval_set=eval_set, verbose=False)
            
            if "calibrated_sigmoid" in cls_method2:
                classifiers[bucket] = CalibratedClassifierCV(classifiers[bucket], cv="prefit", method='sigmoid')
                classifiers[bucket].fit(X_val, y_val)
            elif "calibrated_isotonic" in cls_method2:
                classifiers[bucket] = CalibratedClassifierCV(classifiers[bucket], cv="prefit", method='isotonic')
                classifiers[bucket].fit(X_val, y_val)
            
        
        prefix_lengths_test = dt_test_prefixes.groupby(dataset_manager.case_id_col).size()
        
        # test separately for each prefix length
        for nr_events in range(min_prefix_length, max_prefix_length+1):
            print("Predicting for %s events..."%nr_events)

            # select only cases that are at least of length nr_events
            relevant_cases_nr_events = prefix_lengths_test[prefix_lengths_test == nr_events].index

            if len(relevant_cases_nr_events) == 0:
                break

            dt_test_nr_events = dataset_manager.get_relevant_data_by_indexes(dt_test_prefixes, relevant_cases_nr_events)
            del relevant_cases_nr_events

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

                elif bucket not in classifiers:
                    # use the general class ratio (in training set) as prediction 
                    preds_bucket = [dataset_manager.get_class_ratio(train)] * len(relevant_cases_bucket)

                else:
                    # make actual predictions
                    X_test = feature_combiners[bucket].transform(dt_test_bucket)
                    preds_pos_label_idx = np.where(classifiers[bucket].classes_ == 1)[0][0]
                    preds_bucket = classifiers[bucket].predict_proba(X_test)[:,preds_pos_label_idx]

                preds.extend(preds_bucket)

                # extract actual label values
                test_y_bucket = dataset_manager.get_label_numeric(dt_test_bucket) # one row per case
                test_y.extend(test_y_bucket)
                
                case_ids = list(dt_test_bucket.groupby(dataset_manager.case_id_col).first().index)
                current_results = pd.DataFrame({"dataset": dataset_name, "cls": cls_method2, "params": method_name, "nr_events": nr_events, "predicted": preds_bucket, "actual": test_y_bucket, "case_id": case_ids})
                detailed_results = pd.concat([detailed_results, current_results], axis=0)

            if len(set(test_y)) < 2:
                auc = None
            else:
                auc = roc_auc_score(test_y, preds)
            prec, rec, fscore, _ = precision_recall_fscore_support(test_y, [0 if pred < 0.5 else 1 for pred in preds], average="binary")

            fout.write("%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method2, nr_events, "auc", auc))
            fout.write("%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method2, nr_events, "precision", prec))
            fout.write("%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method2, nr_events, "recall", rec))
            fout.write("%s;%s;%s;%s;%s;%s\n"%(dataset_name, method_name, cls_method2, nr_events, "fscore", fscore))
            
        print("\n")

detailed_results.to_csv(detailed_results_file, sep=";", index=False)