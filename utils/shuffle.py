import random
import numpy as np

from utils import nutils


class ShuffleSpliter(object):

    def __init__(self, n_samples, test_size=0.2):
        '''
        :param n_samples: int, the number of samples
        :param test_size: int or float, if float it's the ratio of test samples.
        '''
        self.n_samples = n_samples

        if test_size < 0.0:
            raise ValueError("the parameter test_size(%.5f) must be int or float which is greater than 0"%test_size)
        if test_size < 1.0:
            self.test_size = self.n_samples*test_size
        else:
            self.test_size = test_size
        self.test_size = int(np.floor(self.test_size))

    def shuffle(self):
        '''
        :return: train indexes and test indexes
        '''
        res = nutils.shuffle(self.n_samples)
        test_ix = res[:self.test_size]
        train_ix = res[self.test_size:]
        return train_ix, test_ix

