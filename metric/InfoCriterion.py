import numpy as np
from collections import Counter
import operator
import sys

from base.CART import SplitInfo
from utils import nutils

Test = False


class InfoCriterion(object):
    def __init__(self):
        pass

    def calc_info(self, y, n_samples, class_dict):
        raise NotImplementedError

    def info(self, y, X=None, discrete=True):
        if X is None:
            return self.info_y(y)
        elif discrete:
            return self.info_discrete(y, X)
        else:
            return self.info_continuous(y, X)

    def info_y(self, y):
        class_dict = dict(Counter([v[0] for v in y]))
        n_samples = len(y)
        return self.calc_info(y, n_samples, class_dict)


    def info_discrete(self, y, X, values=None):
        n_samples = len(X)
        info = 0.0
        if values is None:
            values = np.unique(X)
        values = list(values)
        if isinstance(values[0], SplitInfo):
            infos = []
            for v in values:
                mask = np.in1d(X, v.left)
                left = np.nonzero(mask)[0]
                right = np.nonzero(~mask)[0]
                lpi = len(left)/float(n_samples)
                rpi = len(right)/float(n_samples)
                info = np.dot(lpi, self.info_y(y[left])) + np.dot(rpi, self.info_y(y[right]))
                infos.append(info)
            chosen_index = self.get_best_axis(infos)
            return infos[chosen_index], values[chosen_index]

        else:
            for v in values:
                indices = np.nonzero((X == v))[0]
                if len(indices) == 0:
                    continue
                if Test:
                    print('v %d y %s' % (v, [v[0] for v in y[indices]]))
                pi = len(indices) / float(n_samples)
                info += np.dot(pi, self.info_y(y[indices]))
            return info, np.unique(X)

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
            split = (a+b)/2.0
            left = np.nonzero(X < split)[0]
            right = np.nonzero(X > split)[0]
            lpi = len(left)/float(n_samples)
            rpi = len(right)/float(n_samples)
            info = np.dot(lpi, self.info_y(y[left]))+np.dot(rpi, self.info_y(y[right]))
            infos.append(info)
            splits.append(split)
            a = b
        chosen_index = self.get_best_axis(infos)
        split = splits[chosen_index]
        info = infos[chosen_index]
        return info, split


    def get_info(self, y, X, discrete=True):
        pass

    def get_best_axis(self, infos):
        if isinstance(infos, dict):
            axis = min(infos.items(), key=operator.itemgetter(1))[0]
            return axis
        else:
            return np.asarray(infos).argmin()

    def update_y(self, y):
        self._info = self.info(y)


class Gini(InfoCriterion):
    def __init__(self):
        self.worst = sys.maxsize

    def calc_info(self, y, n_samples, class_dict):
        gi = 1.0
        for key, val in class_dict.items():
            pi = float(val) / n_samples
            gi -= np.power(pi, 2)
        return gi

    def get_info(self, y, X, discrete=True):
        return self.info(y, X, discrete)

class Gain(InfoCriterion):
    def __init__(self):
        self.worst = -10

    def calc_info(self, y, n_samples, class_dict):
        ent = 0.0
        for key, val in class_dict.items():
            pi = float(val)/n_samples
            if pi != 0:
                ent += np.dot(pi, np.log2(pi))
        return -ent

    def get_info(self, y, X, discrete=True):
        enti = self.info(y, X, discrete)
        if Test:
            print('total entropy %.5f entropy on a is %.5f'%(self._info, enti))
        return self._info - enti

    def get_best_axis(self, infos):
        if isinstance(infos, dict):
            axis = max(infos.items(), key=operator.itemgetter(1))[0]
            return axis
        else:
            return np.asarray(infos).argmax()

class MSE(InfoCriterion):
    def __init__(self):
        self.worst = sys.maxsize

    def calc_info(self, y, n_samples, class_dict):
        return np.sqrt(np.sum(y**2)-n_samples*np.mean(y))



if __name__ == '__main__':
    X = np.asarray([1,0,1,1,1,1,0,0,0])
    X = X.reshape((-1, 3))
    print(X)
    y = np.asarray([1,0,1])
    y = y.reshape((-1,1))
    gini = Gini(y)
    print(gini.choose_best_feature(X, y, set()))
    gain = Gain(y)
    print(gain.choose_best_feature(X, y, set()))
    for axis in range(3):
        print('gini %.5f' % gini.get_info(y, X[:,axis]))
        print('gain %.5f' % gain.get_info(y, X[:,axis]))
