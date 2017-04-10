from base import BaseOptimizer
import numpy as np

from utils import nutils


class LassoForwardSelection(BaseOptimizer):
    def solve(self, X, y, params):
        y_ = y.copy()
        n_classes = y_.shape[1]
        for j in range(n_classes):
            features_ = set(list(range(X.shape[1])))
            yj_ = y_[:, j]
            for i in range(X.shape[1]):
                corr = np.abs(nutils.corr(X[:, list(features_)], yj_))
                k = corr.argmax()
                params[k][j] = np.dot(yj_, X[:, k]) / np.sum(X[:, k] ** 2)
                features_ -= {k}
                yj_ = y_[:, j] - np.dot(X, params)[:, j]
        return params


class LassoForwardStagewise(BaseOptimizer):
    def __init__(self, eps=0.01, tol=0.1):
        self.eps = eps
        self.tol = tol
    def solve(self, X, y, params):
        y_ = y.copy()
        n_classes = y_.shape[1]
        for j in range(n_classes):
            yj_ = y_[:, j]
            step = 0
            while np.sum(yj_**2) > self.tol:
                corr = nutils.corr(X, yj_)
                k = corr.argmax()
                _delta = self.eps*np.sign(np.dot(yj_, X[:,k]))
                params[k][j] += _delta
                yj_ -= _delta * X[:, k]

                if step >1000000:
                    print(k)
                    print(_delta*X[:,k])
                    print("delta %.5f yj%s"%(_delta,yj_[:10]))
                step += 1
        return params


class LARS(BaseOptimizer):
    def __init__(self, tol):
        self.tol = tol
    def solve(self, X, y, params):
        pass

    def _AA(self, GA):
        np.sqrt(1/(np.inv(GA)))
    def _GA(self, X):
        return np.dot(X.T, X)