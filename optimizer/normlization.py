import numpy as np


class BaseNormalizer(object):
    def feval(self, params):
        raise NotImplementedError


class ZeroNormalizer(BaseNormalizer):
    def feval(self, params):
        return 0, 0


class L2Normalizer(BaseNormalizer):
    def feval(self, params):
        return np.sum(params ** 2), params
