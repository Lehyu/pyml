import random
import numpy as np

from models.base_estimator import BaseEstimator
class Sampling(object):
    def __init__(self):
        pass

    def sampling(self):
        pass

class BootStrap(Sampling):
    def __init__(self, n_samples):
        self.n_samples = n_samples

    def sampling(self):
        _slice = []
        while len(_slice) < self.n_samples:
            p = random.randrange(0, self.n_samples)
            _slice.append(p)
        return _slice


if __name__ == '__main__':
    boostrap = BootStrap(10)
    for i in range(5):
        _slice = boostrap.sampling()
        print(_slice)