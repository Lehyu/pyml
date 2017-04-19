import xgboost as xgb

from base import BaseEstimator


class SelectFromModel(BaseEstimator):
    def __init__(self, estimator, size=0.3,prefit=False):
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
            self.best_cols_ = [int(item[0][1:]) for item in scores_[:int(len(scores_)*self.size_)]]
        return self

    def transform(self, X, y=None):
        return X[:, self.best_cols_]
