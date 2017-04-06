import queue
from threading import Thread

import numpy as np

from models import BaseEstimator
from models.tree import DecisionTreeClassifier, DecisionTreeRegressor
from preprocessing import BootStrap
from utils import ShuffleSpliter
from utils import logger

Test = False

class BaseForest(BaseEstimator):
    def __init__(self, criterion='gini', n_estimators=10, min_samples_splits=2, min_samples_leaf=1, max_depth=None, n_jobs=1,
                 max_features=None):
        """
        :param n_estimators: the number of estimators
        :param min_samples_splits:
        :param min_samples_leaf:
        :param max_depth: int or None
        :param n_jobs:
        :param max_features: int/float/str, "log2","auto","sqrt"
        """
        self.criterion = criterion
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_splits
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.max_features = max_features
        self.bootstrap = None
        self.producer_index = 0
        self.customer_index = 0
        self.sample_queue = queue.Queue(maxsize=self.n_estimators)
        self.estimators = list()
        if self.n_jobs == -1:
            self.n_jobs = 10

    def _producer(self, n_samples, n_features):
        while self.producer_index < self.n_estimators:
            train_slice = self.bootstrap.sampling()
            val_slice = list(set(range(n_samples)) - set(train_slice))
            spliter = ShuffleSpliter(n_samples=n_features, test_size=self.max_features)
            _, features = spliter.shuffle()
            if Test:
                print(features)
                print('train_slice %s' % train_slice)
                print('val_slice %s' % val_slice)
                print('features %s' % features)
            self.sample_queue.put((train_slice, val_slice, features))
            self.producer_index += 1

    def _customer(self, X, y):
        while self.customer_index < self.n_estimators:
            if self.sample_queue.empty():
                continue
            train_slice, val_slice, features = self.sample_queue.get()
            base_model = self.base_model(criterion=self.criterion, max_depth=self.max_depth,
                                                    min_samples_leaf=self.min_samples_leaf,
                                                    min_samples_split=self.min_samples_split)
            '''
            unchosen_set = set(features)
            base_classifier.n_features = X.shape[1]
            base_classifier.root = base_classifier._build_tree(X[train_slice, :], y[train_slice], unchosen_set, depth=1)
            base_classifier.root = base_classifier._prune_tree(X[val_slice, :], y[val_slice], base_classifier.root)
            '''
            base_model.fit(X[train_slice, :], y[train_slice], train_slice, val_slice)
            self.estimators.append(base_model)
            self.customer_index += 1

    def _check_max_features(self):
        if isinstance(self.max_features, str):
            if self.max_features == "auto":
                self.max_features = self.n_features
            elif self.max_features == "log2":
                self.max_features = int(np.log2(self.n_features))
            elif self.max_features == "sqrt":
                self.max_features = int(np.sqrt(self.n_features))
            else:
                raise ValueError("For now only support sqrt/log2/auto! Please check out!")
        elif isinstance(self.max_features, float):
            self.max_features = int(self.max_features * self.n_features)

    def fit(self, X, y=None):
        self.logger.check(X, y)
        n_samples, n_features = X.shape
        self.n_features = n_features
        if self.max_depth is None:
            self.max_depth = n_features
        self._check_max_features()

        self.bootstrap = BootStrap(n_samples)
        y = y.reshape((-1, 1))
        t_producer = Thread(target=self._producer, args=(n_samples, n_features))
        t_producer.daemon = True
        t_producer.start()
        threads = []
        threads.append(t_producer)
        for i in range(self.n_jobs):
            t_customer = Thread(target=self._customer, args=(X, y))
            t_customer.daemon = True
            t_customer.start()
            threads.append(t_customer)
        for t in threads:
            t.join()
        self.train = True

    def predict(self, X):
        assert self.train, "fitting isn't done!"
        predicts = []
        for x in X:
            predicts.append(self._predict(x))
        predicts = np.asarray(predicts)
        return predicts

    def _predict(self, x):
        predicts = []
        for estimator in self.estimators:
            predicts.append(estimator._predict(x))
        return np.mean(np.asarray(predicts))
class RandomForestRegressor(BaseForest):
    def __init__(self, criterion='gini', n_estimators=10, min_samples_splits=2, min_samples_leaf=1, max_depth=None,
                 n_jobs=1, max_features=None):
        super().__init__(criterion, n_estimators, min_samples_splits, min_samples_leaf, max_depth, n_jobs, max_features)
        self.logger = logger("RandomForestRegressor")
        self.base_model = DecisionTreeRegressor


class RandomForestClassifier(BaseForest):
    def __init__(self, criterion='gini', n_estimators=10, min_samples_splits=2, min_samples_leaf=1, max_depth=None,
                 n_jobs=1, max_features=None):
        super().__init__(criterion, n_estimators, min_samples_splits, min_samples_leaf, max_depth, n_jobs, max_features)
        self.logger = logger("RandomForestClassifier")
        self.base_model = DecisionTreeClassifier