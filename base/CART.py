from collections import Counter

import numpy as np
import itertools


from base.BaseEstimator import BaseEstimator
from base.SplitInfo import SplitInfo
from metric.InfoCriterion import *
from utils import nutils
from utils.logger import logger

Test = False
class TreeNode(object):
    def __init__(self, feat_or_label, left=None, right=None, split=None):
        self.feat_or_label = feat_or_label
        self.left = left
        self.right = right
        self.leaf = False
        if self.left is None and self.right is None:
            self.leaf = True
        self.split = split

class BaseTree(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.logger.check(X, y)
        self.n_features = X.shape[1]
        n_samples = X.shape[0]
        y = y.reshape((n_samples, -1))
        unchosen_set = set(range(self.n_features))
        self.root = self._build_tree(X, y, unchosen_set)
        return self
        pass

    def predict(self, X):

        if self.root is None:
            raise ValueError("Please train the model first!")
        pred = []
        X = X.reshape((-1, self.n_features))
        for x in X:
            pred.append(self._predict(x))
        pred = np.asarray(pred)
        pred = pred.reshape((-1,1))
        return pred

    def _predict(self, X):
        pass

    def _build_tree(self, X, y, unchosen_set):
        if len(np.unique(y)) == 1 or len(unchosen_set) == 1:
            return self._build_leaf(y)
        chosen_axis, split = self._choose_best_features(X, y, unchosen_set)
        if Test:
            print('chosen_axis %d, split %s' % (chosen_axis, split))
        if isinstance(split, SplitInfo):
            mask = np.in1d(X[:, chosen_axis], split.left)
        else:
            mask = X[:, chosen_axis] < split
        if len(X[mask, :]) == 0:
            left = self._build_leaf(y)
        else:
            left = self._build_tree(X[mask, :], y[mask, :], unchosen_set - {chosen_axis})
        if len(X[~mask, :]) == 0:
            right = self._build_leaf(y)
        else:
            right = self._build_tree(X[~mask, :], y[~mask, :], unchosen_set - {chosen_axis})
        root = TreeNode(chosen_axis, left, right, split)
        return root

    def _build_leaf(self, y):
        label = max(Counter([v[0] for v in y]).items(), key=operator.itemgetter(1))[0]
        return TreeNode(label)


    def _choose_best_features(self, X, y, unchosen_set):
        infos = dict()
        splits = dict()
        if Test:
            print('unchosen %d' % len(unchosen_set))
        for axis in unchosen_set:

            discrete = nutils.check_discrete(np.unique(X[:, axis]))
            if discrete:
                if len(np.unique(X[:, axis])) != 1:
                    splitinfos = self.generate_combs(set(np.unique(X[:, axis])))
                    info, splitinfo = self.criterion.info_discrete(y, X[:, axis], splitinfos)
                else:
                    info, splitinfo = self.criterion.worst, self.criterion.worst

            else:
                info, splitinfo = self.criterion.info_continuous(y, X[:, axis])
            infos[axis] = info
            splits[axis] = splitinfo
            # types.append(discrete)
        chosen_axis = self.criterion.get_best_axis(infos)
        return chosen_axis, splits[chosen_axis]

    def generate_combs(self, values):
        n_class = len(values)
        combs = list()
        if n_class == 1:
            combs.append(SplitInfo(values, values - values))
        half = int(n_class / 2)
        for i in range(1, half + 1):
            combs.extend([SplitInfo(set(comb), values - set(comb)) for comb in itertools.combinations(values, i)])

        combs = set(combs)
        return combs

    def _predict(self, x):
        '''
        :param x: (1, n_features)
        :return: pred (1,1)
        '''
        p = self.root
        while not p.leaf:
            if isinstance(p.split, SplitInfo):
                key = x[p.feat_or_label]
                if key in p.split.left:
                    p = p.left
                else:  # key in p.split.right:
                    p = p.right
            else:
                if x[p.feat_or_label] > p.split:
                    p = p.right
                else:
                    p = p.left
        return p.feat_or_label

    def score(self, predict, y_val):
        pass

class DecisionTreeClassifier(BaseTree):

    def __init__(self, criterion='Gini'):
        self.criterion = eval(criterion)()
        self.logger = logger('DecisionTreeClassifier')

    def score(self, predict, y_val):
        _score = 0.0
        for i in range(len(predict)):
            _score += 1 if predict[i] == y_val[i] else 0
        return float(_score)/len(predict)

class DecisionTreeRegressor(BaseTree):
    def __init__(self, criterion='MSE'):
        self.criterion = eval(criterion)()
        self.logger = logger('DecisionTreeRegressor')

    def score(self, predict, y_val):
        y_val = y_val.reshape(predict.shape)
        return np.sqrt(np.sum((predict - y_val)**2))

if __name__ == '__main__':
    tree = DecisionTreeClassifier()
    values = set(range(16))
    combs = tree.generate_combs(values)
    print(combs)