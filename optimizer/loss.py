import numpy as np


class Loss(object):
    def feval(self, X, y, params):
        raise NotImplementedError


class LossWithSumOfSquare(Loss):

    def __init__(self, _lambda):
        self._lambda = _lambda

    def feval(self, X, y, params):
        '''
        :param X: (n_samples, n_features)
        :param y: (n_samples, n_classes)
        :param params: params['coef'] (n_features, n_classes), params['bias'] constant
        :return:
        '''
        assert X.shape[0] == y.shape[0] and X.shape[1] == params['coef'].shape[0] \
               and y.shape[1] == params['coef'].shape[1], \
            "X shape %s y shape %s params['coef'] shape %s"%(X.shape, y.shape, params['coef'].shape)
        n_samples, n_features = X.shape
        n_classes = y.shape[1]
        t = np.dot(X, params['coef']) + params['bias']
        res = y - t
        loss = (np.sum(res**2)/n_samples+self._lambda*np.sum(params['coef']**2)/(n_classes*n_features))/2.0
        coef_grad = (-np.dot(X.T, res)+self._lambda*params['coef'])
        bias_grad = (-np.sum(res))
        return loss, coef_grad, bias_grad
