class logger(object):
    def __init__(self, name):
        self._estimator_name = name

    def check(self, X, y):
        assert len(X) == len(y), "length of X doesn't match the length of y."