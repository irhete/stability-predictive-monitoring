import pandas as pd
import numpy as np
import sys
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from time import time
import pickle
import os
from sys import argv

import EncoderFactory
import BucketFactory
import ClassifierFactory
from DatasetManager import DatasetManager


datasets = ["production", "sepsis_cases_1", "sepsis_cases_2", "sepsis_cases_4", "traffic_fines_1", "bpic2012_accepted",
            "bpic2012_cancelled", "bpic2012_declined", "bpic2017_accepted", "bpic2017_refused", "bpic2017_cancelled",
            "hospital_billing_3"]
outfile = "prefix_lengths_with_classes.csv"

train_ratio = 0.8
val_ratio = 0.2
random_state = 22
    
    
##### MAIN PART ######    
with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s;%s\n"%("dataset", "data_type", "label", "nr_events", "case_count"))
    
    for dataset_name in datasets:
        print(dataset_name)
        dataset_manager = DatasetManager(dataset_name)
        
        # read the data
        data = dataset_manager.read_dataset()
        
        # split data into train and test
        train, test = dataset_manager.split_data_strict(data, train_ratio)
        train, val = dataset_manager.split_val(train, val_ratio)
        
        # consider prefix lengths until 90% of positive cases have finished
        min_prefix_length = 1
        max_prefix_length = dataset_manager.get_max_case_length(data)
        del data

        prefix_lengths_train = train.groupby(dataset_manager.case_id_col).size()
        prefix_lengths_val = val.groupby(dataset_manager.case_id_col).size()
        prefix_lengths_test = test.groupby(dataset_manager.case_id_col).size()
        
        for nr_events in range(min_prefix_length, max_prefix_length+1):
            print(nr_events)
            
            # all counts
            count_train = sum(prefix_lengths_train >= nr_events)
            count_val = sum(prefix_lengths_val >= nr_events)
            count_test = sum(prefix_lengths_test >= nr_events)
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, "train", "all", nr_events, count_train))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, "val", "all", nr_events, count_val))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, "test", "all", nr_events, count_test))
            
            # class counts
            relevant_cases = prefix_lengths_train[prefix_lengths_train >= nr_events].index
            class_counts_train = train[train[dataset_manager.case_id_col].isin(relevant_cases)].groupby(dataset_manager.case_id_col).first()[dataset_manager.label_col].value_counts()
            
            relevant_cases = prefix_lengths_val[prefix_lengths_val >= nr_events].index
            class_counts_val = val[val[dataset_manager.case_id_col].isin(relevant_cases)].groupby(dataset_manager.case_id_col).first()[dataset_manager.label_col].value_counts()
            
            relevant_cases = prefix_lengths_test[prefix_lengths_test >= nr_events].index
            class_counts_test = test[test[dataset_manager.case_id_col].isin(relevant_cases)].groupby(dataset_manager.case_id_col).first()[dataset_manager.label_col].value_counts()
            
            # pos counts
            count_train = 0 if dataset_manager.pos_label not in class_counts_train else class_counts_train[dataset_manager.pos_label]
            count_val = 0 if dataset_manager.pos_label not in class_counts_val else class_counts_val[dataset_manager.pos_label]
            count_test = 0 if dataset_manager.pos_label not in class_counts_test else class_counts_test[dataset_manager.pos_label]
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, "train", "pos", nr_events, count_train))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, "val", "pos", nr_events, count_val))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, "test", "pos", nr_events, count_test))
            
            # neg counts
            count_train = 0 if dataset_manager.neg_label not in class_counts_train else class_counts_train[dataset_manager.neg_label]
            count_val = 0 if dataset_manager.neg_label not in class_counts_val else class_counts_val[dataset_manager.neg_label]
            count_test = 0 if dataset_manager.neg_label not in class_counts_test else class_counts_test[dataset_manager.neg_label]
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, "train", "neg", nr_events, count_train))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, "val", "neg", nr_events, count_val))
            fout.write("%s;%s;%s;%s;%s\n"%(dataset_name, "test", "neg", nr_events, count_test))
            
        print("\n")
