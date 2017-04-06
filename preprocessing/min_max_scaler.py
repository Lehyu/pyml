from preprocessing.base_preprocessor import BasePreprocessor
from utils.logger import logger


class MinMaxScaler(BasePreprocessor):
    """
    X_std = (X-X_min)/(X_max-X_min)
    X_scaled = X_std*(up_bound - low_bound)+low_bound
    """
    def __init__(self, feature_range=(0,1)):
        self.logger = logger("MinMaxScaler")
        self.lb = feature_range[0]
        self.up = feature_range[1]

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def transform(self, X, y=None):
        X_min = self.X_min
        X_max = self.X_max
        X_std = (X - X_min) / (X_max - X_min)
        X_scaled = X_std * (self.up - self.lb) + self.lb
        return X_scaled

    def fit(self, X, y=None):
        self.X_min = X.min(axis=0)
        self.X_max = X.max(axis=0)

