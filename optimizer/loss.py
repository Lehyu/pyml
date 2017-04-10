import numpy as np

from utils import nutils


class Loss(object):
    def feval(self, X, y, params):
        """
        :param X: (n_samples, n_features)
        :param y: (n_samples, n_classes)
        :param params: params['coef'] (n_features, n_classes), params['bias'] constant
        :return:
        """


class LossWithSumOfSquare(Loss):
    def __init__(self, _lambda, normalizer):
        self._lambda = _lambda
        self._normalizer = normalizer

    def feval(self, X, y, params):
        """
        :param X: (n_samples, n_features)
        :param y: (n_samples, n_classes)
        :param params: params['coef'] (n_features, n_classes), params['bias'] constant
        :return:
        """
        '''
        # validation should be checked first before solve.
        assert X.shape[0] == y.shape[0] and X.shape[1] == params['coef'].shape[0] \
               and y.shape[1] == params['coef'].shape[1], \
            "X shape %s y shape %s params['coef'] shape %s" % (X.shape, y.shape, params['coef'].shape)
        '''
        n_samples, n_features = X.shape
        t = np.dot(X, params['coef']) + params['bias']
        res = y - t
        penalty, params_grad = self._normalizer.feval(params['coef'])
        #loss = (np.sum(res ** 2) / n_samples + self._lambda * np.sum(params['coef'] ** 2)) / 2.0
        loss = (np.sum(res ** 2) / n_samples + self._lambda*penalty) / 2.0
        #coef_grad = (-np.dot(X.T, res) + self._lambda * params['coef'])
        coef_grad = -np.dot(X.T, res) + params_grad*self._lambda
        bias_grad = -np.sum(res)
        return loss, coef_grad, bias_grad


class LossWithLogits(Loss):
    def __init__(self, _lambda):
        self._lambda = _lambda

    def feval(self, X, y, params):
        """
        :param X: (n_samples, n_features)
        :param y: (n_samples, 1)
        :param params: params['coef'] (n_features, 1), params['bias'] constant
        :return:
        """
        n_samples, n_features = X.shape
        h = nutils.sigmoid(np.dot(X, params['coef']) + params['bias'])
        res = h - y
        loss = -np.sum((y * np.log2(h) + (1 - y) * np.log2(1 - h)).clip(1e-10, 0.9999999)) / n_samples + \
               self._lambda * np.sum(params['coef'] ** 2) / 2.0
        coef_grad = np.dot(X.T, res) + self._lambda * params['coef']
        bias_grad = np.sum(res)
        return loss, coef_grad, bias_grad


class LossWithSoftmax(Loss):
    def __init__(self, _lambda):
        self._lambda = _lambda

    def feval(self, X, y, params):
        n_samples, n_features = X.shape
        h = nutils.sofmax(np.dot(X, params['coef']) + params['bias'])
        res = h - y
        loss = -np.sum((y * np.log2(h)).clip(1e-10, 0.9999999)) / n_samples + self._lambda * np.sum(
            params['coef'] ** 2) / 2.0
        coef_grad = np.dot(X.T, res) + self._lambda * params['coef']
        bias_grad = np.sum(res)
        return loss, coef_grad, bias_grad
