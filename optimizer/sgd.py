import sys

from base import BaseOptimizer
from .loss import LossWithSumOfSquare, LossWithLogits, LossWithSoftmax
from utils import nutils

LOSS = {'SumOfSquares': LossWithSumOfSquare,
        'LossWithLogits': LossWithLogits,
        'LossWithSoftmax': LossWithSoftmax}
Test = False


class SGD(BaseOptimizer):
    def __init__(self, learning_rate=1e-1, eps=1e-5, max_iter=100000, batch_size=10, loss="SumOfSquares", decay='step',
                 _lambda=0.1):
        self.learning_rate = learning_rate
        self.eps = eps
        self.max_iter = max_iter
        self.batch_size = batch_size
        self._loss = LOSS[loss](_lambda)

    def solve(self, X, y, params):
        """
        :param X: (n_samples, n_features)
        :param y: (n_samples, n_classes)
        :param params: (n_features, n_classes)
        :param feval: target function
        :return:
        """
        n_samples, n_features = X.shape
        loss = sys.maxsize
        epoch = 0
        learning_rate = self.learning_rate
        while loss > self.eps and epoch < self.max_iter:
            total_loss = 0.0
            for batch in nutils.batch(n_samples, self.batch_size):
                loss, coef_grad, bias_grad = self._loss.feval(X[batch], y[batch], params)
                total_loss += loss * len(batch)
                params['coef'] -= learning_rate * coef_grad
                params['bias'] -= learning_rate * bias_grad
            total_loss /= n_samples
            # todo
            if self._check_convergence():
                break
            epoch += 1
            learning_rate = self._tune_learning_rate(learning_rate, epoch)
            if epoch % 100 == 0 and Test:
                print("epoch %d, loss %.5f" % (epoch, total_loss))
        return params

    def _tune_learning_rate(self, learning_rate, epoch):

        return learning_rate
