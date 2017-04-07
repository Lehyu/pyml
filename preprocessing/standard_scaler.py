import numpy as np

from base import BasePreprocessor

'''
For instance many elements used in the objective function of a learning algorithm
(such as the RBF kernel of Support Vector Machines or the L1 and L2 regularizers of linear models)
assume that all features are centered around 0 and have variance in the same order.
quoted from scikit-learn.
'''


class StandardScaler(BasePreprocessor):
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def fit(self, X, y=None):
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)

    def transform(self, X, y=None):
        X_mean = self.X_mean
        X_std = self.X_std
        return (X - X_mean) / X_std
