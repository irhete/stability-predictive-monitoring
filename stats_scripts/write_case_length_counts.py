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
outfile = "case_lengths_with_classes.csv"

    
##### MAIN PART ######    
with open(outfile, 'w') as fout:
    
    fout.write("%s;%s;%s;%s\n"%("dataset", "label", "nr_events", "case_count"))
    
    for dataset_name in datasets:
        print(dataset_name)
        dataset_manager = DatasetManager(dataset_name)
        
        # read the data
        data = dataset_manager.read_dataset()
        
        prefix_lengths = data[data[dataset_manager.label_col]==dataset_manager.pos_label].groupby(dataset_manager.case_id_col).size().value_counts()
        for length, count in prefix_lengths.iteritems():
            fout.write("%s;%s;%s;%s\n"%(dataset_name, "pos", length, count))
        prefix_lengths = data[data[dataset_manager.label_col]!=dataset_manager.pos_label].groupby(dataset_manager.case_id_col).size().value_counts()
        for length, count in prefix_lengths.iteritems():
            fout.write("%s;%s;%s;%s\n"%(dataset_name, "neg", length, count))
        