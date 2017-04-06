from models.base_estimator import BaseEstimator


class BaseSearchCV(BaseEstimator):
    def __init__(self, estimator, param_grid, scoring, ):
        pass
    def fit(self, X, y=None):
        pass


class GridSearchCV():
    pass