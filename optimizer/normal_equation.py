import numpy as np

from optimizer.base_optimizer import BaseOptimizer


class NormalEquation(BaseOptimizer):
    def solve(self, X, y, params, feval):
        """
        :param X: (n_samples, n_features)
        :param y: (n_samples, n_classes)
        :param params: (n_features, n_classes)
        :param feval: target function
        :param ef: error function
        :return: params (n_features, n_classes)
        """
        return np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)


