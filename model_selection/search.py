import itertools
import numpy as np
from collections import defaultdict

from base import BaseEstimator
from model_selection.split import RandomSpliter


class BaseSearchCV(BaseEstimator):
    def __init__(self, estimator, params_grid, scorer, post_operator=None, cv=None, verbose=False, pre_split=True):
        """
        :param estimator: scikit-learn regressor/classifier class not model
        :param params_grid: params_grid to fit estimator
        :param scorer:  scorer must callable
        :param cv: if None, then must add validation dataset in fit
        """
        self.estimator = estimator
        self.params_grid = params_grid
        self.scorer = scorer
        self.cv = cv
        self.post_operator = post_operator
        self.verbose = verbose
        self.best_params_ = None
        self.best_score_ = self.scorer.worst
        self.cv_spliter = None
        self.pre_split = pre_split

    def _fit(self, X_train, y_train=None, X_val=None, y_val=None, addition_y=None):
        raise NotImplementedError

    def fit(self, X_train, y_train=None, X_val=None, y_val=None, addition_y=None):
        if self.cv is not None:
            if X_val is not None:
                n_samples = np.r_[X_train, X_val].shape[0]
            else:
                n_samples = X_train.shape[0]
            self.cv_spliter = RandomSpliter(n_samples, self.cv, self.pre_split)
        self._fit(X_train, y_train, X_val, y_val, addition_y)

    def _score(self, model, X_train, y_train, X_val, y_val, addition_y=None):
        clf = model
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        if self.post_operator is not None and addition_y is not None:
            y_pred = self.post_operator.post_process(y_pred, addition_y)
            y_val = self.post_operator.post_process(y_val, addition_y)
        elif self.post_operator is not None and addition_y is None:
            raise ValueError("addition_y should not be None!")
        # print(y_pred-y_val)
        score = self.scorer.score(y_val, y_pred)
        assert score >= self.scorer.tol, "score(%5.f) should be greater than %.5f. " \
                                         "please check out the features that feed in!" % (score, self.scorer.tol)
        return score

    def _scoreCV(self, model, X_train, y_train, addition_y):
        score = 0.0
        cv = 1
        for train_ix, val_ix in self.cv_spliter.split():
            if self.post_operator is None:
                s = self._score(model, X_train[train_ix], y_train[train_ix], X_train[val_ix], y_train[val_ix])
            else:
                s = self._score(model, X_train[train_ix], y_train[train_ix], X_train[val_ix], y_train[val_ix],
                                addition_y=addition_y[val_ix])
            if self.verbose and self.__class__.__name__ == "SingleModelSearchCV":
                print("CV%d....... score%.5f" % (cv, s))
            cv += 1
            score += s
        score /= self.cv
        return score

    def _check(self, model, X_train, y_train, X_val, y_val, addition_y=None):
        if self.cv is not None:
            score = self._scoreCV(model, X_train, y_train, addition_y)
        else:
            score = self._score(model, X_train, y_train, X_val, y_val, addition_y)
        return self.scorer.better(score, self.best_score_), score

    def _update(self, _params, model, X_train, y_train, X_val, y_val, addition_y=None):
        model.set_params(**_params)
        flag, score = self._check(model, X_train, y_train, X_val, y_val, addition_y)
        if flag:
            self.best_score_ = score
            self.best_params_ = _params
            if self.verbose:
                print("%s score %.5f" % (_params, score))


class SingleModelSearchCV(BaseSearchCV):
    def __init__(self, estimators, scorer, post_operator=None, cv=None, verbose=None):
        self.estimators = estimators
        self.scorer = scorer
        self.post_operator = post_operator
        self.cv = cv
        self.verbose = verbose
        self.scores = defaultdict()
        self.best_score_ = self.scorer.worst
        self.pre_split = True

    def _fit(self, X_train, y_train=None, X_val=None, y_val=None, addition_y=None):
        if self.cv is None:
            for _id, estimator in enumerate(self.estimators):
                if self.post_operator is not None:
                    score = self._score(estimator, X_train, y_train, X_val, y_val, addition_y=addition_y)
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
            if X_val is not  None:
                X_train = np.r_[X_train, X_val]
                y_train = np.r_[y_train, y_val]
            for train_ix, val_ix in self.cv_spliter.split():
                for _id, estimator in enumerate(self.estimators):
                    if self.post_operator is not None:
                        score = self._score(estimator, X_train[train_ix], y_train[train_ix], X_train[val_ix],
                                            y_train[val_ix], addition_y=addition_y[val_ix])
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
                self.scores[key] /= self.cv

    def get_scores(self):
        return self.scores


class GridSearchCV(BaseSearchCV):
    def __init__(self, estimator, params_grid, scorer, post_operator=None, params_seq=None, cv=None, verbose=False):
        """
        :param estimator:
        :param params_grid:
        :param scorer:
        :param post_operator:
        :param params_seq: a list of tuple, a tuple is what you think that may has a high correlation.
        :param cv:
        :param verbose:
        """
        super(GridSearchCV, self).__init__(estimator=estimator, params_grid=params_grid, scorer=scorer,
                                           post_operator=post_operator, cv=cv, verbose=verbose)
        self.params_seq = params_seq
        if self.params_seq is None:
            self.params_seq = list(self.params_grid.keys())
        self.best_params_ = dict()
        for keys in self.params_seq:
            for key in keys:
                self.best_params_[key] = self.estimator().get_params()[key]

    def _fit(self, X_train, y_train=None, X_val=None, y_val=None, addition_y=None):
        for keys in self.params_seq:
            tmp_params_grid_ = dict()
            for key in keys:
                tmp_params_grid_[key] = self.params_grid[key]
            keys = tmp_params_grid_.keys()
            for vals in itertools.product(*tmp_params_grid_.values()):
                _params = self.best_params_.copy()
                for key, val in dict(zip(keys, vals)).items():
                    _params[key] = val
                model = self.estimator()
                self._update(_params, model, X_train, y_train, X_val, y_val, addition_y)


class FullSearchCV(BaseSearchCV):
    def __init__(self, estimator, params_grid, scorer, post_operator=None, cv=None, verbose=False):
        super(FullSearchCV, self).__init__(estimator, params_grid, scorer, post_operator, cv, verbose)

    def _fit(self, X_train, y_train=None, X_val=None, y_val=None, addition_y=None):
        model = self.estimator()
        keys = self.params_grid.keys()
        _params = dict()
        for key in keys:
            _params[key] = model.get_params()[key]
        self._update(_params, model, X_train, y_train, X_val, y_val, addition_y)
        for vals in itertools.product(*self.params_grid.values()):
            _params = dict(zip(keys, vals))
            model = self.estimator()
            self._update(_params, model, X_train, y_train, X_val, y_val, addition_y)
