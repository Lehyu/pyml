import operator
import sys
from collections import Counter

import numpy as np

from models.tree.split_info import SplitInfo
from utils import nutils


class InfoCriterion(object):
    def __init__(self):
        pass

    def calc_info(self, y, n_samples, class_dict):
        raise NotImplementedError

    def info_y(self, y):
        class_dict = dict(Counter([v[0] for v in y]))
        n_samples = len(y)
        return self.calc_info(y, n_samples, class_dict)

    def info_discrete(self, y, X, values=None):
        if isinstance(values[0], SplitInfo):
            return self._info_binary(y, X, values)
        else:
            return self._info_not_binary(y, X)

    def _info_binary(self, y, X, values):
        n_samples = len(X)
        infos = []
        for v in values:
            mask = np.in1d(X, v.left)
            left = np.nonzero(mask)[0]
            right = np.nonzero(~mask)[0]
            lpi = len(left) / float(n_samples)
            rpi = len(right) / float(n_samples)
            info = np.dot(lpi, self.info_y(y[left])) + np.dot(rpi, self.info_y(y[right]))
            infos.append(info)
        chosen_index = self.get_best_axis(infos)
        return infos[chosen_index], values[chosen_index]

    def _info_not_binary(self, y, X):
        n_samples = len(X)
        info = 0.0
        values = np.unique(X)
        for v in values:
            mask = (X == v)
            v_samples = len(X[mask])
            if v_samples == 0:
                continue
            pi = v_samples / float(n_samples)
            info += np.dot(pi, self.info_y(y[mask]))
        return info, values

    def info_continuous(self, y, X):
        values = np.unique(X)
        if len(values) == 0:
            return
        if len(values) == 1:
            return np.dot(1.0, self.info_y(y)), values[0]
        sorted(values)
        a = values[0]
        n_samples = len(X)
        infos = []
        splits = []
        for ix in range(1, len(values)):
            b = values[ix]
            split = (a + b) / 2.0
            mask = X < split
            lpi = len(X[mask]) / float(n_samples)
            rpi = len(X[~mask]) / float(n_samples)
            info = np.dot(lpi, self.info_y(y[mask])) + np.dot(rpi, self.info_y(y[~mask]))
            infos.append(info)
            splits.append(split)
            a = b
        chosen_index = self.get_best_axis(infos)
        return infos[chosen_index], splits[chosen_index]

    def get_info(self, y, X, discrete=True):
        pass

    def get_best_axis(self, infos):
        if isinstance(infos, dict):
            if len(infos) == 0:
                axis = None
            else:
                axis = min(infos.items(), key=operator.itemgetter(1))[0]
            return axis
        else:
            return np.asarray(infos).argmin()

    def update_y(self, y):
        self._info = self.info_y(y)

class gini(InfoCriterion):
    def __init__(self):
        self.worst = sys.maxsize

    def calc_info(self, y, n_samples, class_dict):
        gi = 1.0
        for key, val in class_dict.items():
            pi = float(val) / n_samples
            gi -= np.power(pi, 2)
        return gi


class gain(InfoCriterion):
    def __init__(self):
        self.worst = -10

    def calc_info(self, y, n_samples, class_dict):
        ent = 0.0
        for key, val in class_dict.items():
            pi = float(val) / n_samples
            ent += nutils.shannon_ent(pi)
        return -ent

    def get_best_axis(self, infos):
        if isinstance(infos, dict):
            axis = max(infos.items(), key=operator.itemgetter(1))[0]
            return axis
        else:
            return np.asarray(infos).argmax()


class mse(InfoCriterion):
    def __init__(self):
        self.worst = sys.maxsize

    def calc_info(self, y, n_samples, class_dict):
        info = np.sum((y-np.mean(y))**2)
        return info

