
class Estimator(object):

    def fit(self, X, y=None):
        raise NotImplementedError

    def transform(self):
        pass

    def predict(self, X):
        pass

    def fit_transform(self, X, y=None):
        pass

