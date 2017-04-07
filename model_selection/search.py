import itertools

from base import BaseEstimator


class BaseSearchCV(BaseEstimator):
    def __init__(self, estimator, params_grid, scorer, post_operator=None, params_seq=None, cv=None, verbose=False):
        """
        :param estimator: scikit-learn regressor/classifier class not model
        :param params_grid: params_grid to fit estimator
        :param scorer:  scorer must callable
        :param params_seq: if None, perform params randomly
        :param cv: if None, then must add validation dataset in fit
        """
        self.estimator = estimator
        self.params_grid = params_grid
        self.scorer = scorer
        self.params_seq = params_seq
        self.cv = cv
        self.post_operator = post_operator
        self.verbose = verbose
        if self.params_seq is None:
            self.params_seq = list(self.params_grid.keys())
        self.best_params_ = self.estimator().get_params()
        keys_to_remove = set(self.best_params_.keys())-set(self.params_seq)
        for key in keys_to_remove:
            self.best_params_.pop(key)
        self.best_score_ = self.scorer.worst

    def _fit(self, X_train, y_train=None, X_val=None, y_val=None, addition_y=None):
        raise NotImplementedError

    def fit(self, X_train, y_train=None, X_val=None, y_val=None, addition_y=None):
        self._fit(X_train, y_train, X_val, y_val, addition_y)

    def _score(self, model, X_train, y_train, X_val, y_val, addition_y=None):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        if self.post_operator is not None and addition_y is not  None:
            y_pred = self.post_operator.post_process(y_pred, addition_y)
            y_val = self.post_operator.post_process(y_val, addition_y)
        elif self.post_operator is not  None and addition_y is None:
            raise ValueError("addition_y should not be None!")
        score = self.scorer.score(y_val, y_pred)
        return score

    def _scoreCV(self, model, X_train, y_train):
        pass

    def _update(self, _params, model, X_train, y_train, X_val, y_val, addition_y=None):
        model.set_params(**_params)
        if self.cv is None:
            score = self._score(model, X_train, y_train, X_val, y_val, addition_y)
        else:
            score = self._scoreCV(model, X_train, y_train)
        if self.verbose:
            print("%s score %.5f"%(_params, score))
        if self.scorer.better(score, self.best_score_):
            self.best_score_ = score
            self.best_params_ = _params


class GridSearchCV(BaseSearchCV):
    def __init__(self, estimator, params_grid, scorer, post_operator=None, params_seq=None, cv=None, verbose=False):
        super(GridSearchCV, self).__init__(estimator, params_grid, scorer, post_operator, params_seq, cv, verbose)

    def _fit(self, X_train, y_train=None, X_val=None, y_val=None, addition_y=None):
        for key in self.params_seq:
            for val in self.params_grid[key]:
                model = self.estimator()
                _params = self.best_params_.copy()
                _params[key] = val
                self._update(_params, model, X_train, y_train, X_val, y_val, addition_y)


class FullSearchCV(BaseSearchCV):
    def __init__(self, estimator, params_grid, scorer, post_operator=None, params_seq=None, cv=None, verbose=False):
        super(FullSearchCV, self).__init__(estimator, params_grid, scorer, post_operator, params_seq, cv, verbose)


    def _fit(self, X_train, y_train=None, X_val=None, y_val=None, addition_y=None):
        keys = self.params_grid.keys()
        for vals in itertools.product(*self.params_grid.values()):
            _params = dict(zip(keys, vals))
            model = self.estimator()
            model.set_params(**_params)
            self._update(_params, model, X_train, y_train, X_val, y_val, addition_y)
