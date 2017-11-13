import itertools
import threading
from threading import Thread
from queue import Queue

import numpy as np
from collections import defaultdict

import time

from base import BaseEstimator
from utils import safeutils


class BaseSearchCV(BaseEstimator):
    def __init__(self, estimator, params_grid, scorer, proba=False,cv_spliter=None, n_jobs=50, verbose=False):
        """
        :param estimator: scikit-learn regressor/classifier class not model
        :param params_grid: params_grid to fit estimator
        :param scorer:  scorer must callable
        :param cv: if None, then must add validation dataset in fit
        :param post_operator:
        :param cv_spliter:
        :param n_jobs
        :param verbose
        """
        self.estimator = estimator
        self.params_grid = params_grid
        self.scorer = scorer
        self.verbose = verbose
        self.best_params_ = None
        self.best_score_ = self.scorer.worst
        self.cv_spliter = cv_spliter
        self.n_jobs = n_jobs
        self.params_queue = Queue(maxsize=1000)
        self.mutex = threading.Lock()
        self.index = 1
        self.produced = False
        self.proba = proba

    def _params_producer(self, params_combs, keys):
        index = 0
        while index < len(params_combs):
            self.params_queue.put(dict(zip(keys, params_combs[index])))
            index += 1
        print("producer done...")
        self.produced = True

    def _params_customer(self,X,y):
        while True:
            if self.produced and self.params_queue.empty():
                break
            params = self.params_queue.get()
            if params is None:
                print("params is None")
            model = self.estimator(**params)
            score = self._scoreCV(model, X, y)
            self.mutex.acquire()
            self.index += 1
            if self.scorer.better(score, self.best_score_):
                self.best_score_ = score
                self.best_params_ = params
                if self.verbose:
                    print("%s score %.5f" % (params, score))
            self.mutex.release()
            self.params_queue.task_done()
            if self.index % 100 == 0:
                print("finished %d search" % self.index)

    def _fit(self, X, y):
        raise NotImplementedError

    def fit(self, X, y=None):
        safeutils.apply_sklearn_workaround()
        self._fit(X, y)

    def _score(self, clf, X_train, y_train, X_val, y_val):
        clf.fit(X_train, y_train)
        if self.proba:
            y_pred = clf.predict_proba(X_val)
            if y_pred.shape[1] == 2:
                y_pred = y_pred[:,-1]
        else:
            y_pred = clf.predict(X_val)
        score = self.scorer.score(y_val, y_pred)
        assert score >= self.scorer.tol, "score(%5.f) should be greater than %.5f. " \
                                         "please check out the features that feed in!" % (score, self.scorer.tol)
        return score

    def _scoreCV(self, model, X, y):
        score = 0.0
        cv = 1
        for train_ix, val_ix in self.cv_spliter.split(X):
            s = self._score(model, X[train_ix], y[train_ix], X[val_ix], y[val_ix])
            if self.verbose and self.__class__.__name__ == "SingleModelSearchCV":
                print("CV%d....... score%.5f" % (cv, s))
            cv += 1
            score += s
        score /= self.cv_spliter.n_splits
        return score

    def _check(self, model, X_train, y_train, X_val, y_val):
        if self.cv_spliter is not None:
            score = self._scoreCV(model, X_train, y_train)
        else:
            score = self._score(model, X_train, y_train, X_val, y_val)
        return score


class SingleModelSearchCV(BaseSearchCV):
    def __init__(self, estimators, scorer, estimator, params_grid, proba=False, post_operator=None, cv_spliter=None, verbose=None):
        super().__init__(estimator, params_grid, scorer, proba, post_operator, cv_spliter, verbose)
        self.estimators = estimators
        self.scorer = scorer
        self.post_operator = post_operator
        self.verbose = verbose
        self.scores = defaultdict()
        self.best_score_ = self.scorer.worst
        self.cv_spliter = cv_spliter

    def _fit(self, X_train, y_train=None, X_val=None, y_val=None, addition_y=None):
        if self.cv_spliter is not None:
            for _id, estimator in enumerate(self.estimators):
                if self.post_operator is not None:
                    score = self._score(estimator, X_train, y_train, X_val, y_val)
                else:
                    score = self._score(estimator, X_train, y_train, X_val, y_val)
                key = str(_id) + '._' + estimator.__class__.__name__
                if key in self.scores:
                    self.scores[key] += score
                else:
                    self.scores[key] = score
                if self.verbose:
                    print("estimator:%s.....score:%.5f" % (key, score))
        else:
            cv = 1
            print(self.estimators)
            if X_val is not None:
                X_train = np.r_[X_train, X_val]
                y_train = np.r_[y_train, y_val]
            for train_ix, val_ix in self.cv_spliter.split(X_train):
                for _id, estimator in enumerate(self.estimators):
                    if self.post_operator is not None:
                        score = self._score(estimator, X_train[train_ix], y_train[train_ix], X_train[val_ix],
                                            y_train[val_ix])
                    else:
                        score = self._score(estimator, X_train[train_ix], y_train[train_ix], X_train[val_ix],
                                            y_train[val_ix])
                    key = str(_id) + '._' + estimator.__class__.__name__
                    print(key)
                    if key in self.scores:
                        self.scores[key] += score
                    else:
                        self.scores[key] = score
                    if self.verbose:
                        print("CV%d......estimator:%s.....score:%.5f" % (cv, key, score))
                cv += 1
            for key in self.scores.keys():
                self.scores[key] /= self.cv_spliter.cv

    def get_scores(self):
        return self.scores


class GridSearchCV(BaseSearchCV):
    def __init__(self, estimator, params_grid, scorer, proba=False, params_seq=None, cv_spliter=None, verbose=False, n_jobs=1):
        super(GridSearchCV, self).__init__(estimator=estimator, params_grid=params_grid, scorer=scorer, proba=proba,
                                          cv_spliter=cv_spliter, verbose=verbose, n_jobs=n_jobs)
        self.params_seq = params_seq
        if self.params_seq is None:
            self.params_seq = list(self.params_grid.keys())
        self.best_params_ = dict()
        for keys in self.params_seq:
            for key in keys:
                self.best_params_[key] = self.estimator().get_params()[key]

    def _fit(self, X, y=None):
        for keys in self.params_seq:
            tmp_params_grid_ = dict()
            for key in self.params_grid.keys():
                if key in keys:
                    tmp_params_grid_[key] = self.params_grid[key]
                else:
                    tmp_params_grid_[key] = [self.best_params_[key]]
            keys = tmp_params_grid_.keys()
            params_combs = list(itertools.product(*(tmp_params_grid_.values())))
            t_params_producer = Thread(target=self._params_producer, args=(params_combs, keys))
            t_params_producer.daemon = True
            threads = [t_params_producer]
            for i in range(self.n_jobs):
                t_params_customer = Thread(target=self._params_customer, args=(X, y))
                t_params_customer.daemon = True
                threads.append(t_params_customer)

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()


class FullSearchCV(BaseSearchCV):
    def __init__(self, estimator, params_grid, scorer, proba=False, cv_spliter=None, n_jobs=50, verbose=False):
        super().__init__(estimator, params_grid, scorer, proba, cv_spliter, n_jobs, verbose)

    def _fit(self, X, y):
        keys = self.params_grid.keys()
        params_combs = list(itertools.product(*(self.params_grid.values())))
        params_num = len(params_combs)
        t_params_producer = Thread(target=self._params_producer, args=(params_combs, keys))
        t_params_producer.daemon = True
        t_params_producer.start()
        threads = [t_params_producer]
        while self.params_queue.qsize() < self.params_queue.maxsize / 2 and self.params_queue.qsize() < params_num / 2:
            time.sleep(1)
        for i in range(self.n_jobs):
            t_params_customer = Thread(target=self._params_customer, args=(X, y))
            t_params_customer.daemon = True
            t_params_customer.start()
            threads.append(t_params_customer)

        for thread in threads:
            thread.join()
