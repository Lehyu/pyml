import numpy as np


class Kernel(object):
    def transform(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples), dtype=np.float64)
        for ix in range(n_samples):
            K[ix, :] = self._kernel_transform(X, X[ix])
        return K

    def _kernel_transform(self, X, x):
        raise NotImplementedError


class RBFKernel(Kernel):
    def __init__(self, sigma):
        self.sigma = sigma

    def _kernel_transform(self, X, x):
        n_size = X.shape[0]
        k = np.zeros((1, n_size),dtype=np.float64)
        for ix in range(n_size):
            delta = X[ix] - x
            k[:, ix] = np.dot(delta, delta)
        k = np.exp(k/ (-2.0 * self.sigma ** 2))
        return k


class PolyKernel(Kernel):
    def __init__(self, degree=3):
        self.degree = degree

    def _kernel_transform(self, X, x):
        return (np.dot(X, x) + 1) ** self.degree


class LinearKernel(Kernel):
    def __init__(self, constant):
        self.constant = constant

    def _kernel_transform(self, X, x):
        return np.dot(X, x) + self.constant
