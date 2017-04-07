import itertools
from collections import Counter
import operator
import numpy as np

from metric.info_criterion import gini, mape, mse
from base import BaseEstimator
from base import SplitInfo
from utils import nutils
from utils import logger
from utils import ShuffleSpliter
from metric import score

#Test = True
Test = False


class TreeNode(object):
    def __init__(self, feat_or_label, depth, memory, left=None, right=None, split=None):
        self.feat_or_label = feat_or_label
        self.left = left
        self.right = right
        self.leaf = False
        self.depth = depth
        self.memory = memory
        if self.left is None and self.right is None:
            self.leaf = True
        self.split = split

    def __str__(self):
        fm = "feat_or_label: %.5f \n leaf: %s \n depth: %d \n split: %.5f"
        return fm % (self.feat_or_label, self.leaf, self.depth, self.split)


class BaseTree(BaseEstimator):
    def __init__(self, criterion='gini', max_depth=None, test_size=0.2, min_samples_split=2, min_samples_leaf=2):
        """
        :param criterion: string, gini, entropy, mse(default='gini')
        :param max_depth:  int or None, the maximum of the tree
        :param test_size:  int or float, split a part of samples to prune the tree in order to ease the over-fitting
        :param min_samples_split: int or float. the minimum number of samples required to split
        :param min_samples_leaf: the minimum number of leaf
        """
        self.min_samples_split = min_samples_split
        self.criterion = eval(criterion)()
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.test_size = test_size
        self.logger = None
        self.root = None
        self.discrete = []

    def fit(self, X, y=None, train_ix=None, test_ix=None):
        self.logger.check(X, y)
        self.n_features = X.shape[1]
        '''
        if self.max_depth == None:
            self.max_depth = self.n_features
        '''
        n_samples = X.shape[0]
        y = y.reshape((n_samples, ))
        spliter = ShuffleSpliter(n_samples, test_size=self.test_size)
        if train_ix is None:
            train_ix, test_ix = spliter.shuffle()
        unchosen_set = set(range(self.n_features))
        for axis in unchosen_set:
            discrete = nutils.check_discrete(np.unique(X[:, axis]))
            self.discrete.append(discrete)
        self.root = self._build_tree(X[train_ix, :], y[train_ix], unchosen_set, depth=1)
        self.root = self._prune_tree(X[test_ix, :], y[test_ix], self.root)
        return self

    def predict(self, X):
        assert self.root is not None, "Please train the model first!"
        pred = []
        X = X.reshape((-1, self.n_features))
        for x in X:
            pred.append(self._predict(x))
        pred = np.asarray(pred)
        pred = pred.reshape((-1,))
        return pred

    def _predict(self, X):
        pass

    def _build_tree(self, X, y, unchosen_set, depth):
        # if all samples have the same label, then mark it as leaf
        if len(np.unique(y)) == 1:
            return self._build_leaf(y, depth)

        # pre-prune
        if 0< len(y) < self.min_samples_leaf  or \
                (self.max_depth is not None and depth == self.max_depth-1):# or len(X) < 2*self.min_samples_split:
            return self._build_leaf(y, depth)


        # choose split point
        chosen_axis, split = self._choose_best_features(X, y, unchosen_set)
        if chosen_axis is None:
            return self._build_leaf(y, depth)
        if Test:
            print('chosen_axis %d, split %s' % (chosen_axis, split))
        # discrete or continuous
        if isinstance(split, SplitInfo):
            mask = np.in1d(X[:, chosen_axis], split.left)
        else:
            mask = X[:, chosen_axis] < split
        '''
        if len(X[mask, :]) == 0:
            left = self._build_leaf(y, depth + 1)
        '''
        if len(unchosen_set-{chosen_axis}) == 0 or len(X[mask,:]) == 0 or len(X[~mask, :]) == 0:
            left = self._build_leaf(y[mask], depth) if len(X[mask, :]) != 0 else None
            right = self._build_leaf(y[~mask], depth) if len(X[~mask, :]) != 0 else None
        else:
            left = self._build_tree(X[mask, :], y[mask], unchosen_set - {chosen_axis}, depth + 1)
            right = self._build_tree(X[~mask, :], y[~mask], unchosen_set - {chosen_axis}, depth + 1)
        root = TreeNode(chosen_axis, depth, self._counter(y), left, right, split)
        return root

    def _build_leaf(self, y, depth):
        y_counts = self._counter(y)
        label = self._get_label(y_counts)
        return TreeNode(label, depth, y_counts)

    def _counter(self, y):
        if len(y.shape) == 2:
            y_counts = Counter([v[0] for v in y])
        else:
            y_counts = Counter([v for v in y])
        return y_counts

    def _prune_tree(self, X, y, node):
        if not node.leaf:
            # prune left
            if node.left is not None:
                left = node.left
                self._prune_tree(X, y, left)
                score_before_prune = self._score(y, self.predict(X))
                node.left = self._branch2leaf(left)
                score_after_prue = self._score(y, self.predict(X))
                if not self._compare(score_before_prune, score_after_prue):
                    node.left = left

                elif Test:
                    print('improve from %.5f to %.5f' % (score_before_prune, score_after_prue))
            # prune right
            if node.right is not None:
                right = node.right
                self._prune_tree(X, y, right)
                score_before_prune = self._score(y, self.predict(X))
                node.right = self._branch2leaf(right)
                score_after_prue = self._score(y, self.predict(X))
                if not self._compare(score_before_prune, score_after_prue):
                    node.right = right
                elif Test:
                    print('improve from %.5f to %.5f' % (score_before_prune, score_after_prue))
        return node

    def _branch2leaf(self, node):
        label = self._get_label(node.memory)
        return TreeNode(label, node.depth, node.memory)

    def _choose_best_features(self, X, y, unchosen_set):
        infos = dict()
        splits = dict()
        if Test:
            print('unchosen %d' % len(unchosen_set))
        for axis in unchosen_set:
            if self.discrete[axis]:
                if len(np.unique(X[:, axis])) > 1:
                    splitinfos = self.generate_combs(set(np.unique(X[:, axis])))
                    '''
                    for splitinfo in list(splitinfos):
                        mask = np.in1d(X[:, axis], splitinfo.left)
                        if len(X[mask]) < self.min_samples_split or len(X[~mask]) < self.min_samples_split:
                            splitinfos.remove(splitinfo)
                    if len(splitinfos) == 0:
                        continue
                    '''
                    info, splitinfo = self.criterion.info_discrete(y, X[:, axis], list(splitinfos))
                else:
                    continue
                    #info = self.criterion.info_y(y)
                    #splitinfo = self.criterion.worst
            else:
                info, splitinfo = self.criterion.info_continuous(y, X[:, axis])

            infos[axis] = info
            splits[axis] = splitinfo
            # types.append(discrete)
        if Test:
            print("unchosen set %s infos %s"%(unchosen_set, infos))
        chosen_axis = self.criterion.get_best_axis(infos)
        split = None
        if chosen_axis is not None:
            split = splits[chosen_axis]
        return chosen_axis, split

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
        """
        :param x: (1, n_features)
        :return: pred (1,1)
        """
        p = self.root
        while not p.leaf:
            # print('feat/label %d x value %.5f split %.5f' % (p.feat_or_label, x[p.feat_or_label], p.split))
            if (isinstance(p.split, SplitInfo) and (x[p.feat_or_label] in p.split.left)) \
                    or (not isinstance(p.split, SplitInfo) and x[p.feat_or_label] < p.split):
                if p.left is None:
                    return self._get_label(p.memory)
                p = p.left
            else:
                if p.right is None:
                    return self._get_label(p.memory)
                p = p.right
        # print('predict %.5f'%p.feat_or_label)
        return p.feat_or_label

    def _get_label(self, y):
        pass

    def _compare(self, score_before_prune, score_after_prue):
        pass

    def _score(self, predict, y_val):
        pass


class DecisionTreeClassifier(BaseTree):
    def __init__(self, criterion='gini', max_depth=None, test_size=0.2, min_samples_split=2, min_samples_leaf=1):
        super().__init__(criterion, max_depth, test_size, min_samples_split, min_samples_leaf)
        self.logger = logger('DecisionTreeClassifier')

    def _get_label(self, y_counts):
        label = max(y_counts.items(), key=operator.itemgetter(1))[0]
        return label

    def _score(self, y, y_pred):
        return score.accuracy(y, y_pred)

    def _compare(self, score_before_prune, score_after_prue):
        return score_before_prune < score_after_prue


class DecisionTreeRegressor(BaseTree):
    def __init__(self, criterion='mse', max_depth=None, test_size=0.2, min_samples_split=2, min_samples_leaf=1):
        super().__init__(criterion, max_depth, test_size, min_samples_split, min_samples_leaf)
        self.logger = logger('DecisionTreeRegressor')

    def _get_label(self, y_counts):
        total = 0.0
        number = 0
        for val, num in y_counts.items():
            total += val * num
            number += num
        return total / float(number)

    def _score(self, y, y_pred):
        return score.MAPE(y, y_pred)

    def _compare(self, score_before_prune, score_after_prue):
        return score_before_prune > score_after_prue


if __name__ == '__main__':
    tree = DecisionTreeClassifier()
    values = set(range(16))
    combs = tree.generate_combs(values)
    print(combs)
