from base import BaseEstimator


class BasePreprocessor(BaseEstimator):
    def fit(self, X, y=None):
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        raise NotImplementedError

    def transform(self, X, y=None):
        raise NotImplementedError
