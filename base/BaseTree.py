import operator
from collections import Counter

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from base.BaseEstimator import BaseEstimator
from metric.InfoCriterion import *
from utils.logger import logger


class TreeNode(object):
    def __init__(self, val, subtrees=None, discrete=True, split=None):
        self.val = val
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
        self.root = self._build_tree(X, y, set())
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
        print(x)
        p = self.root
        while not p.leaf:
            if p.discrete:
                p = p.subtrees[x[p.val]]
            else:
                if x[p.val] > p.split:
                    p = p.subtrees[operator.gt]
                else:
                    p = p.subtrees[operator.lt]
        return p.val

    def _choose_features(self, X, y, chosen_set):
        self.criterion.update_y(y)
        chosen_axis = self.criterion.choose_best_feature(X, y, chosen_set)
        return chosen_axis
    # RecursionError: maximum recursion depth exceeded in comparison
    def _build_tree(self, X, y, chosen_set):
        if len(np.unique(y)) == 1:
            # mark this node as np.unique(y)
            return TreeNode(np.unique(y)[0])
        n_samples, n_features = X.shape
        unchosen_set = set(range(n_features)) - chosen_set
        if len(unchosen_set) == 1:
            # mark this node as most(y)
            C = max(Counter([v[0] for v in y]).items(), key=operator.itemgetter(1))[0]
            node = TreeNode(C)
            return node
        chosen_axis, values, discrete = self._choose_features(X, y, chosen_set)
        chosen_set.add(chosen_axis)
        subtrees = {}
        if discrete:
            for val in values:
                indices = X[:, chosen_axis] == val
                subtrees[val] = self._build_tree(X[indices,:], y[indices,:], chosen_set)
            root = TreeNode(chosen_axis, subtrees, discrete)
        else:
            split = values
            left = X[:, chosen_axis] < split
            right = X[:, chosen_axis] > split
            subtrees[operator.lt] = self._build_tree(X[left, :], y[left, :], chosen_set)
            subtrees[operator.gt] = self._build_tree(X[right, :], y[right, :], chosen_set)
            root = TreeNode(chosen_axis, subtrees, discrete, split)
        return root



if __name__ == '__main__':
    '''
    X = np.asarray([1, 0, 1, 1, 1, 1, 0, 0, 0])
    X = X.reshape((-1, 3))
    y = np.asarray([1, 0, 1])
    y = y.reshape((-1, 1))
    print(X)
    print(y)

    tree = BaseTree(criterion='Gini')
    tree = tree.fit(X, y)
    print(tree.predict(X))
    '''
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    print(X_train.shape)
    print(y_train.shape)
    tree = BaseTree(criterion='Gini')
    tree = tree.fit(X_train, y_train)
    pred = tree.predict(X_val)
    print([v for v in y_val])
    print(pred)








