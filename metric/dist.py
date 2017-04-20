import numpy as np

Criterions = ("Euclidean",)

class Dist(object):
    def __init__(self, criterion):
        if not criterion in Criterions:
            raise ValueError("We don't support %s yet" % self.criterion)
        self.criterion = criterion

    def dist(self, x, y):
        return self.feval(x, y)

    def feval(self, x, y):
        if self.criterion == Criterions[0]:
            return self._Euclidean(x, y)

    def _Euclidean(self, x, y):
        return np.mean((x-y)**2)

    def _Minkowski(self, x, y, p):
        pass
