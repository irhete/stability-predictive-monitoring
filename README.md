# Temporal Stability in Predictive Process Monitoring
This repository contains the code for the experiments conducted in the article ["Temporal stability in predictive process monitoring"](https://link.springer.com/article/10.1007/s10618-018-0575-9) by [Irene Teinemaa](https://irhete.github.io/), [Marlon Dumas](http://kodu.ut.ee/~dumas/), [Anna Leontjeva](https://scholar.google.com/citations?user=XkCYSbQAAAAJ&hl=fr), and [Fabrizio Maria Maggi](https://scholar.google.nl/citations?user=Jo9fNKEAAAAJ&hl=en&oi=sra), published in Data Mining and Knowledge Discovery, as part of the Journal Track of ECML PKDD 2018.

The repository contains scripts for outcome-oriented predictive business process monitoring (i.e. classification models for complex sequences):

* training predictive models based on random forest, XGBoost, and LSTM classifiers;
* evaluating prediction accuracy and temporal prediction stability;
* hyperparameter optimization via random search;
* applying exponential smoothing to reduce volatility in consecutive predictions.

11 out of 12 evaluation datasets (labeled and preprocessed) can be found [here](https://drive.google.com/open?id=1a4RClJgmsyrQgCz_1O51gut_N1XoNBhn).


## Reference
If you use the code from this repository, please cite the original paper:
```
@Article{Teinemaa2018,
  title="Temporal stability in predictive process monitoring",
  author={Teinemaa, Irene and Dumas, Marlon and Leontjeva, Anna and Maggi, Fabrizio Maria},
  journal="Data Mining and Knowledge Discovery",
  year="2018",
  month="Jun",
  day="29",
  issn="1573-756X",
  doi="10.1007/s10618-018-0575-9",
  url="https://doi.org/10.1007/s10618-018-0575-9"
} 
```

## Requirements   
The code is written in Python 3.6. Although not tested, it should work with any version of Python 3. Additionally, the following Python libraries are required to run the code: 

* sklearn
* numpy
* pandas
* xgboost
* keras
* theano/tensorflow

Note that the LSTM experiments are computationally expensive, so it is recommended to run them on a GPU.


## Usage
#### Data format
The scripts assume that each input dataset is a CSV file, each row representing one event in a trace, wherein each event is associated with at least the following attributes (as columns in the CSV file): the case id, activity type, timestamp, class label. As additional columns, any number of event and case attributes is accepted that are used to enhance the predictive power of the classifier. The relevant columns for each dataset should be specified in the script `dataset_confs.py`.

The input log is temporally split into data for training (80% of cases) and evaluating (20% of cases) the predictive models. The training data is further split (via random sampling over traces) into a dataset for training the base classifier (64% of all cases in the event log) and a two-purpose validation dataset (16% of all cases in the event log) that is used for selecting the best parameters and for probability calibration. After training and calibrating the models, prediction accuracy and temporal prediction stability are evaluated on the evaluation set.

#### 1. Hyperparameter optimization
The hyperparameters of the random forest, XGBoost, and LSTM models are tuned using random search, i.e. for each dataset and method, 16 randomly chosen parameter configurations are tested (on a validation set) and the configuration that yields the highest AUC is chosen.

1.1. Testing different parameter configurations.

In order to launch experiments on 16 random parameter configurations for each method and dataset, run:

`python random_search.py`   

This script makes use of two other scripts, `python experiments_param_optim_rf_xgboost.py` and `python experiments_param_optim_lstm.py`, which take as input one specific parameter configuration and train and evaluate a predictive model on that setting. The `random_search.py` script assumes a SLURM queue management system. 

1.2. Selecting best parameters.

After the experiments launched via `random_search.py` have finished, the best parameters can be extracted by running the following three scripts:

`python extract_best_params_single_run_rf_xgboost.py` 

`python extract_best_params_multiple_runs_rf_xgboost.py` 

`python extract_best_params_lstm.py` 

#### 2. Training, calibrating, and evaluating the (final) models

After extracting the best parameters, the final models can be trained and applied using:

`python experiments_final_rf_xgboost.py <dataset> <method> <classifier> <params_dir> <results_dir>` 

`python experiments_final_lstm.py <dataset> <method> <classifier> <params_dir> <results_dir>` 

The arguments to the scripts are: 

* _dataset_ - the name of the dataset, should correspond to the settings specified in `dataset_confs.py`.
* _method_ - combination of a bucketing and a sequence encoding technique, e.g. _single_agg_. For LSTM, use _lstm_.
* _classifier_ - the name of the classifier. One of: _rf_, _xgboost_, _lstm_. For calibrating the classifier, add _\_calibrated_ to the name, i.e. _rf\_calibrated_, _xgboost\_calibrated_, _lstm\_calibrated_.
* _params_dir_ - the name of the directory where the optimal parameters have been saved (see above).
* _results_dir_ - the name of the directory where the predictions will be written.

The evaluation of the models is done in `plot_results_stability.R`.

#### 3. Exponential smoothing

Exponential smoothing is applied in `plot_results_stability.R`.