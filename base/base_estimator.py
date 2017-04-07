
class BaseEstimator(object):
    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self, X, y=None):
        pass

    def predict(self, X):
        pass

    def fit_transform(self, X, y=None):
        pass

