import threading
import random
from queue import Queue
from threading import Thread

import xgboost as xgb

from base import BaseEstimator
from utils import safeutils


class SelectFromModel(BaseEstimator):
    def __init__(self, estimator, size=0.3, prefit=False):
        self.estimator = estimator
        self.prefit = prefit
        self.best_cols_ = []
        self.size_ = size


    def fit(self, X, y=None):
        if not self.prefit:
            self.estimator.fit(X, y)
        if isinstance(self.estimator, xgb.XGBRegressor):
            scores_ = self.estimator.booster().get_score(importance_type='weight')
            scores_ = sorted(scores_.items(), key=lambda item: item[1], reverse=True)
            self.best_cols_ = [int(item[0][1:]) for item in scores_[:int(len(scores_) * self.size_)]]
        return self

    def transform(self, X, y=None):
        return X[:, self.best_cols_]


class FFS(object):
    def __init__(self, estimator, scorer, n_features_to_select=None, n_jobs=10):
        self.estimator = estimator
        self.scorer = scorer
        self.n_features_to_select = n_features_to_select
        self.best_cols_ = dict()
        self.best_score_ = None
        self.record_cols_ = dict()
        self.cols_queue = Queue(maxsize=500)
        self.produced = False
        self.n_jobs = n_jobs
        self.mutex = threading.Lock()

    def fit(self, tX, ty, vX, vy, original_features=None):
        if self.n_features_to_select is None:
            self.n_features_to_select = int(tX.shape[1]/2)
        self._fit(tX, ty, vX, vy, original_features)

    def _producer(self, curset, remainset):
        for col in remainset:
            newset = curset | {col}
            self.cols_queue.put(newset)
        self.produced = True

    def _customer(self, tX, ty, vX, vy):
        while True:
            if self.produced and self.cols_queue.empty():
                break
            curset = list(self.cols_queue.get())

            model = self.estimator()
            model.fit(tX[:, curset], ty)
            py = model.predict(vX[:, curset])
            score = self.scorer.score(vy, py)
            self.mutex.acquire()
            self.record_cols_[tuple(curset)] = score
            self.mutex.release()
            self.cols_queue.task_done()

    def _fit(self, tX, ty, vX, vy, original_features):

        curset = set() if original_features is None else set(original_features)
        iter = len(curset)
        wholeset = set(range(tX.shape[1]))
        safeutils.apply_sklearn_workaround()
        reverse = self.scorer.greater_is_better
        while iter < self.n_features_to_select:
            remainset = wholeset - curset
            t_params_producer = Thread(target=self._producer, args=(curset, remainset))
            t_params_producer.daemon = True
            threads = [t_params_producer]

            for i in range(self.n_jobs):
                t_params_customer = Thread(target=self._customer, args=(tX, ty, vX, vy))
                t_params_customer.daemon = True
                threads.append(t_params_customer)

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.mutex.acquire()
            scores_ = sorted(self.record_cols_.items(), key=lambda item: item[1], reverse=reverse)
            curset = set(scores_[0][0])
            self.record_cols_.clear()
            self.best_cols_[scores_[0][0]] = scores_[0][1]
            self.mutex.release()
            iter+=1
            print(iter)
        scores_ = sorted(self.best_cols_.items(), key=lambda item: item[1], reverse=reverse)
        self.best_cols_.clear()
        self.best_cols_= list(scores_[0][0])
        self.best_score_ = scores_[0][1]

class LVW(object):
    def __init__(self, estimator, scorer):
        self.estimator = estimator
        self.scorer = scorer
        self.best_score_ = None
        self.best_cols_ = None

    def fit(self, tX, ty, vX, vy, T,original_features):
        E = self.scorer.worst
        A = list(set(range(tX.shape[1]))-set(original_features))
        t = 0
        d = len(original_features)
        while t < T:
            selected_features = list(set(random.sample(A, random.randint(1, len(A)))) | set(original_features))
            d_ = len(selected_features)
            model = self.estimator()
            model.fit(tX[:, selected_features], ty)
            py = model.predict(vX[:, selected_features])
            E_ = self.scorer.score(vy, py)
            if self.scorer.better(E_, E) or (E_ == E and d_ < d):
                t = 0
                E = E_
                d = d_
                self.best_cols_ = selected_features
                self.best_score_ = E_
            else:
                t += 1









