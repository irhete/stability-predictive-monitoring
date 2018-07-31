import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

try:
    from keras.preprocessing.sequence import pad_sequences
except ImportError:
    print("Could not load CUDA")
    
import dataset_confs

class DatasetManager:
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        
        self.case_id_col = dataset_confs.case_id_col[self.dataset_name]
        self.activity_col = dataset_confs.activity_col[self.dataset_name]
        self.timestamp_col = dataset_confs.timestamp_col[self.dataset_name]
        self.label_col = dataset_confs.label_col[self.dataset_name]
        self.pos_label = dataset_confs.pos_label[self.dataset_name]
        self.neg_label = dataset_confs.neg_label[self.dataset_name]

        self.dynamic_cat_cols = dataset_confs.dynamic_cat_cols[self.dataset_name]
        self.static_cat_cols = dataset_confs.static_cat_cols[self.dataset_name]
        self.dynamic_num_cols = dataset_confs.dynamic_num_cols[self.dataset_name]
        self.static_num_cols = dataset_confs.static_num_cols[self.dataset_name]
        
        self.sorting_cols = [self.timestamp_col, self.activity_col]
        
        self.scaler = None
        self.encoded_cols = None
    
    def read_dataset(self):
        # read dataset
        dtypes = {col:"object" for col in self.dynamic_cat_cols+self.static_cat_cols+[self.case_id_col, self.label_col, self.timestamp_col]}
        for col in self.dynamic_num_cols + self.static_num_cols:
            dtypes[col] = "float"

        data = pd.read_csv(dataset_confs.filename[self.dataset_name], sep=";", dtype=dtypes)
        data[self.timestamp_col] = pd.to_datetime(data[self.timestamp_col])

        return data
    
    def split_data_strict(self, data, train_ratio, split="temporal"):  
        # split into train and test using temporal split and discard events that overlap the periods
        data = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind='mergesort')
        train_ids = list(start_timestamps[self.case_id_col])[:int(train_ratio*len(start_timestamps))]
        train = data[data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        test = data[~data[self.case_id_col].isin(train_ids)].sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        split_ts = test[self.timestamp_col].min()
        train = train[train[self.timestamp_col] < split_ts]
        return (train, test)
    
    def split_val(self, data, val_ratio, split="random", seed=22):  
        # split into train and test using temporal split
        grouped = data.groupby(self.case_id_col)
        start_timestamps = grouped[self.timestamp_col].min().reset_index()
        if split == "temporal":
            start_timestamps = start_timestamps.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
        elif split == "random":
            np.random.seed(seed)
            start_timestamps = start_timestamps.reindex(np.random.permutation(start_timestamps.index))
        val_ids = list(start_timestamps[self.case_id_col])[-int(val_ratio*len(start_timestamps)):]
        val = data[data[self.case_id_col].isin(val_ids)].sort_values(self.sorting_cols, ascending=True, kind="mergesort")
        train = data[~data[self.case_id_col].isin(val_ids)].sort_values(self.sorting_cols, ascending=True, kind="mergesort")
        return (train, val)

    def generate_prefix_data(self, data, min_length, max_length):
        # generate prefix data (each possible prefix becomes a trace)
        data['case_length'] = data.groupby(self.case_id_col)[self.activity_col].transform(len)

        dt_prefixes = data[data['case_length'] >= min_length].groupby(self.case_id_col).head(min_length)
        dt_prefixes["prefix_nr"] = 1
        dt_prefixes["orig_case_id"] = dt_prefixes[self.case_id_col]
        for nr_events in range(min_length+1, max_length+1):
            tmp = data[data['case_length'] >= nr_events].groupby(self.case_id_col).head(nr_events)
            tmp["orig_case_id"] = tmp[self.case_id_col]
            tmp[self.case_id_col] = tmp[self.case_id_col].apply(lambda x: "%s_%s"%(x, nr_events))
            tmp["prefix_nr"] = nr_events
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)
        
        dt_prefixes['case_length'] = dt_prefixes.groupby(self.case_id_col)[self.activity_col].transform(len)
        
        return dt_prefixes

    def get_pos_case_length_quantile(self, data, quantile=0.90):
        return int(np.ceil(data[data[self.label_col]==self.pos_label].groupby(self.case_id_col).size().quantile(quantile)))
    
    def get_max_case_length(self, data):
        return data[data[self.label_col]==self.pos_label].groupby(self.case_id_col).size().max()

    def get_indexes(self, data):
        return data.groupby(self.case_id_col).first().index

    def get_relevant_data_by_indexes(self, data, indexes):
        return data[data[self.case_id_col].isin(indexes)]

    def get_label(self, data):
        return data.groupby(self.case_id_col).first()[self.label_col]
    
    def get_case_ids(self, data, nr_events=1):
        case_ids = pd.Series(data.groupby(self.case_id_col).first().index)
        if nr_events > 1:
            case_ids = case_ids.apply(lambda x: "_".join(x.split("_")[:-1]))
        return case_ids
    
    def get_label_numeric(self, data):
        y = self.get_label(data) # one row per case
        return [1 if label == self.pos_label else 0 for label in y]
    
    def get_class_ratio(self, data):
        class_freqs = data[self.label_col].value_counts()
        return class_freqs[self.pos_label] / class_freqs.sum()
    
    def get_prefix_lengths(self, data):
        return data.groupby(self.case_id_col).last()["prefix_nr"]
    
    def encode_data_for_lstm(self, data):
        data = data.sort_values(self.sorting_cols, ascending=True, kind='mergesort')
        
        num_cols = self.dynamic_num_cols + self.static_num_cols
        cat_cols = self.dynamic_cat_cols + self.static_cat_cols
        
        # scale numeric cols
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            dt_all = pd.DataFrame(self.scaler.fit_transform(data[num_cols]), index=data.index, columns=num_cols)
        else:
            dt_all = pd.DataFrame(self.scaler.transform(data[num_cols]), index=data.index, columns=num_cols)
            
        # one-hot encode categorical cols
        dt_cat = pd.get_dummies(data[cat_cols])
        
        # merge
        dt_all = pd.concat([dt_all, dt_cat], axis=1)
        dt_all[self.case_id_col] = data[self.case_id_col]
        dt_all[self.label_col] = data[self.label_col].apply(lambda x: 1 if x == self.pos_label else 0)
        dt_all[self.timestamp_col] = data[self.timestamp_col]
        
        # add missing columns if necessary
        if self.encoded_cols is None:
            self.encoded_cols = dt_all.columns
        else:
            for col in self.encoded_cols:
                if col not in dt_all.columns:
                    dt_all[col] = 0
        
        return dt_all[self.encoded_cols]
    
    def generate_3d_data(self, data, max_len):
        data = data.sort_values(self.timestamp_col, ascending=True, kind="mergesort").groupby(self.case_id_col).head(max_len)
        grouped = data.sort_values(self.timestamp_col, ascending=True, kind="mergesort").groupby(self.case_id_col)

        data_dim = data.shape[1] - 3
        n_cases = data.shape[0]
        
        X = np.zeros((n_cases, max_len, data_dim), dtype=np.float32)
        y = np.zeros((n_cases, 2), dtype=np.float32)

        idx = 0
        # each prefix will be a separate instance
        for _, group in grouped:
            group = group.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
            label = group[self.label_col].iloc[0]
            group = group.as_matrix()
            for i in range(1, len(group) + 1):
                X[idx] = pad_sequences(group[np.newaxis,:i,:-3], maxlen=max_len, dtype=np.float32)
                y[idx, label] = 1
                idx += 1
        return (X, y)
    
    def generate_3d_data_for_prefix_length(self, data, max_len, nr_events):
        grouped = data.groupby(self.case_id_col)
        data_dim = data.shape[1] - 3
        n_cases = np.sum(grouped.size() >= nr_events)
        
        # encode only prefixes of this length
        X = np.zeros((n_cases, max_len, data_dim), dtype=np.float32)
        y = np.zeros((n_cases, 2), dtype=np.float32)
        case_ids = []
        
        idx = 0
        for case_id, group in grouped:
            if len(group) < nr_events:
                continue
            group = group.sort_values(self.timestamp_col, ascending=True, kind="mergesort")
            label = group[self.label_col].iloc[0]
            group = group.as_matrix()
            X[idx] = pad_sequences(group[np.newaxis,:nr_events,:-3], maxlen=max_len, dtype=np.float32)
            y[idx, label] = 1
            case_ids.append(case_id)
            idx += 1

        return (X, y, case_ids)
    