import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA as skPCA

from base import BaseEstimator


class PCA(BaseEstimator):
    """
    This method is just for study and understand how PCA works.
    """

    def __init__(self, M=10, **kwargs):
        self.M = M

    def fit(self, X, y=None):
        n_size, n_features = X.shape
        S = np.cov(X.T)
        eigvals, eigvectors = np.linalg.eig(S)
        eigvectors = linalg.orth(eigvectors)
        indices = np.argsort(eigvals)[-self.M:][::-1]
        self.n_compents = eigvectors[indices]
        self.n_compents = self.n_compents.reshape((n_features, -1))
        self.eigvals = eigvals[indices]
        total = np.sum(eigvals)
        self.compent_ratos = self.eigvals / total

    def transform(self, X, y=None):
        return np.dot(X, self.n_compents)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def reverse(self, X):
        return np.dot(X, self.n_compents.T)


if __name__ == '__main__':
    X = np.array([1, 3, 2, 3, 1, 0, 1, 4, 2, 3, 4, 6], dtype=np.float64)
    X = X.reshape((-1, 3))
    mypca = PCA(2)
    mypca.fit(X)
    X_tran = mypca.transform(X)
    skpac = skPCA(n_components=2)
    skpac.fit(X)
