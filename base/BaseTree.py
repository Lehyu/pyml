import operator
from collections import Counter
from collections import defaultdict
import queue

import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split

from base.BaseEstimator import BaseEstimator
from metric.InfoCriterion import *
from utils.logger import logger


class TreeNode(object):
    def __init__(self, val, subtrees=None, discrete=True, split=None):
        '''
        :param val: label if self.leaf else feature
        :param subtrees: {val:subtree}
        :param discrete:
        :param split:
        '''
        self.feature_or_label = val
        self.subtrees = subtrees
        self.leaf = False
        self.discrete = discrete
        self.split = split
        if subtrees is None:
            self.leaf = True


class BaseTree(BaseEstimator):

    def __init__(self, criterion='Gain'):
        self.criterion = eval(criterion)()
        self.root = None
        self.logger = logger('BaseTree')

    def fit(self, X, y=None):
        self.logger.check(X, y)
        self.n_features = X.shape[1]
        n_samples = X.shape[0]
        y = y.reshape((n_samples, -1))
        unchosen_set = set(range(self.n_features))
        self.root = self._build_tree(X, y, unchosen_set)
        return self

    def predict(self, X):

        if self.root is None:
            raise ValueError("Please train the model first!")
        pred = []
        X = X.reshape((-1, self.n_features))
        for x in X:
            pred.append(self._predict(x))
        return pred

    def _predict(self, x):
        '''
        :param x: (1, n_features)
        :return: pred (1,1)
        '''
        p = self.root
        while not p.leaf:
            if p.discrete:
                key = x[p.feature_or_label]
                if key in p.subtrees:
                    p = p.subtrees[x[p.feature_or_label]]
                else:
                    return self._find_most(p)
            else:
                if x[p.feature_or_label] > p.split:
                    p = p.subtrees[operator.gt]
                else:
                    p = p.subtrees[operator.lt]
        return p.feature_or_label

    def _find_most(self, p):
        labels = []
        nodeQueue = queue.Queue(500)
        nodeQueue.put(p)
        while not nodeQueue.empty():
            p = nodeQueue.get()
            for subp in p.subtrees.values():
                if subp.leaf:
                    labels.append(subp.feature_or_label)
                else:
                    nodeQueue.put(subp)
        return max(Counter(labels).items(), key=operator.itemgetter(1))[0]

    def _choose_best_feature(self, X, y, unchosen_set):
        self.criterion.update_y(y)
        infos = dict()
        values = dict()
        types = dict()
        for axis in unchosen_set:
            discrete = True
            for val in np.unique(X[:, axis]):
                discrete &= nutils.is_int(val)
            info, value = self.criterion.get_info(y, X[:, axis], discrete)
            infos[axis] = info
            values[axis] = value
            types[axis] = discrete
        chosen_axis = self.criterion.get_best_axis(infos)
        return chosen_axis, values[chosen_axis], types[chosen_axis]

    # RecursionError: maximum recursion depth exceeded in comparison
    def _build_tree(self, X, y, unchosen_set):
        if len(np.unique(y)) == 1:
            # mark this node as np.unique(y)
            return self._build_leaf(y)
        if len(unchosen_set) == 1:
            return self._build_leaf(y)
        chosen_axis, values, discrete = self._choose_best_feature(X, y, unchosen_set)
        subtrees = {}
        if discrete:
            for val in values:
                indices = np.nonzero(X[:, chosen_axis] == val)[0]
                if len(indices) == 0:
                    subtrees[val] = self._build_leaf(y)
                else:
                    subtrees[val] = self._build_tree(X[indices,:], y[indices,:], unchosen_set - {chosen_axis})
            root = TreeNode(chosen_axis, subtrees, discrete)
        else:
            split = values
            left = X[:, chosen_axis] < split
            right = X[:, chosen_axis] > split
            for key, indices in {operator.lt: left, operator.gt: right}.items():
                if len(indices) == 0:
                    subtrees[key] = self._build_leaf(y)
                else:
                    subtrees[key] = self._build_tree(X[indices,:], y[indices], unchosen_set-{chosen_axis})
            root = TreeNode(chosen_axis, subtrees, discrete, split)
        return root

    def _build_leaf(self, y):
        C = max(Counter([v[0] for v in y]).items(), key=operator.itemgetter(1))[0]
        return TreeNode(C)

def test_sklearn_data(data):
    X, y = data.data, data.target
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    tree = BaseTree(criterion='Gini')
    tree.fit(X_train, y_train)

    pred = tree.predict(X_val)
    total = 0
    for i in range(len(pred)):
        if pred[i] == y_val[i]:
            total += 1
    print(total/float(len(pred)))
if __name__ == '__main__':
    '''
    X = np.asarray([1, 0, 1, 1, 1, 1, 0, 0, 0])
    X = X.reshape((-1, 3))
    y = np.asarray([1, 0, 1])
    y = y.reshape((-1, 1))

    tree = BaseTree(criterion='Gini')
    tree = tree.fit(X, y)
    print(tree.predict(X))
    '''
    test_sklearn_data(sklearn.datasets.load_iris())