from abc import abstractmethod


class BaseOptimizer(object):
    @abstractmethod
    def solve(self, X, y, params):
        raise NotImplementedError

    def _check_convergence(self):
        pass

