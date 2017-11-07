from sklearn.base import BaseEstimator
import numpy as np

class LSTM2D(BaseEstimator):
    def __init__(self, model, time_dim, n_features):
        self.model = model
        self.time_dim = time_dim
        self.n_features = n_features
    
    
    def predict_proba(self, X):
        return self.model.predict(X.reshape((X.shape[0], self.time_dim, self.n_features)), verbose=0)