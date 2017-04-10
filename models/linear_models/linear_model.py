import numpy as np

from base import BaseEstimator
from optimizer import LossWithLogits
from optimizer import LossWithSoftmax
from optimizer import LossWithSumOfSquare
from optimizer import SGD
from optimizer.lars import LassoForwardSelection, LassoForwardStagewise
from optimizer.normlization import L2Normalizer, ZeroNormalizer
from utils import logger, nutils

LOSS = {'SumOfSquares': LossWithSumOfSquare,
        'LossWithLogits': LossWithLogits,
        'LossWithSoftmax': LossWithSoftmax}


class LinearModel(BaseEstimator):
    def __init__(self):
        self.optimizer = None
        self.logger = None

    def _fit(self, X, y):
        n_samples, n_features = X.shape
        y = y.reshape((n_samples, -1))
        n_classes = y.shape[1]
        self.params["coef"] = np.random.uniform(-0.1, 0.1, (n_features, n_classes))
        self.params["bias"] = np.zeros(1)
        self.params = self.optimizer.solve(X, y, self.params)

    def fit(self, X, y=None):
        self.logger.check(X, y)
        self._fit(X, y)
        self.trained = True


class LinearRegression(LinearModel):
    def __init__(self, learning_rate=1e-1, eps=1e-5, max_iter=1000, batch_size=10, decay='step'):
        loss = LOSS["SumOfSquares"](0, ZeroNormalizer())
        self.optimizer = SGD(learning_rate=learning_rate, eps=eps, max_iter=max_iter, batch_size=batch_size, loss=loss,
                             decay=decay)
        # self.optimizer = NormalEquation()
        self.logger = logger("LinearRegression")
        self.params = dict()
        self.trained = False

    def predict(self, X):
        assert self.trained, "Please fit the model first!"
        return (np.dot(X, self.params['coef']) + self.params['bias']).reshape((X.shape[0]))


class Ridge(LinearModel):
    def __init__(self, learning_rate=1e-1, eps=1e-5, max_iter=1000, batch_size=10, decay='step', reg_lambda=0.1):
        loss = LOSS["SumOfSquares"](reg_lambda, L2Normalizer())
        self.optimizer = SGD(learning_rate=learning_rate, eps=eps, max_iter=max_iter, batch_size=batch_size, loss=loss,
                             decay=decay)
        self.logger = logger("Ridge")
        self.params = dict()
        self.trained = False

    def predict(self, X):
        assert self.trained, "Please fit the model first!"
        return (np.dot(X, self.params['coef']) + self.params['bias']).reshape((X.shape[0]))


class Lasso(LinearModel):
    def __init__(self):
        self.logger = logger("Lasso")
        self.optimizer = LassoForwardSelection()
        self.params = None
        self.X_mean_ = None
        self.y_mean_ = None
        self.X_L2_ = None

    def _fit(self, X, y):
        self.X_mean_ = np.mean(X, axis=0)
        self.y_mean_ = np.mean(y, axis=0)
        X_ = X - self.X_mean_
        y_ = y - self.y_mean_
        self.X_L2_ = np.sqrt(np.sum(X_**2, axis=0))
        X_ /= self.X_L2_
        n_samples, n_features = X.shape
        y_ = y_.reshape((n_samples, -1))
        n_classes = y_.shape[1]
        self.params = np.zeros((n_features, n_classes))
        self.params = self.optimizer.solve(X_, y_, self.params)

    def predict(self, X):
        X_ = X - self.X_mean_
        X_ /= self.X_L2_
        pred = np.dot(X_, self.params) + self.y_mean_
        if pred.shape[1] == 1:
            pred = pred.reshape((-1,))
        return pred


class LogisticRegression(LinearModel):
    def __init__(self, learning_rate=1e-3, eps=1e-5, max_iter=1000, batch_size=10, loss="LossWithSoftmax", decay='step',
                 _lambda=0.1):
        self.loss = LOSS[loss](_lambda, L2Normalizer())
        self.optimizer = SGD(learning_rate=learning_rate, eps=eps, max_iter=max_iter, batch_size=batch_size, loss=loss,
                             decay=decay)
        self.logger = logger("LogisticRegression")
        self.params = dict()
        self.trained = False

    def predict(self, X):
        if self.loss == "LossWithLogits":
            pred = [1 if p > 0.5 else 0 for p in
                    nutils.sigmoid(np.dot(X, self.params['coef']) + self.params['bias']).reshape((X.shape[0]))]
            return np.asarray(pred)
        elif self.loss == "LossWithSoftmax":
            pred = nutils.sofmax(np.dot(X, self.params['coef']) + self.params['bias'])
            return pred.argmax(axis=1)


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.linear_model import LinearRegression as SKLR
    from sklearn.linear_model import LogisticRegression as SKLGR
    from sklearn.model_selection import train_test_split
    from utils import sklutils
    from metric import score as score

    # '''
    mylr = Lasso()
    # mylr = Ridge()
    sklr = SKLR()

    sklutils.compare(sklr, mylr, datasets.load_diabetes(), score.metric, test_size=0.2)
    # '''
    '''
    mylgr = LogisticRegression()
    sklgr = SKLGR(multi_class='multinomial', solver="lbfgs")
    for i in range(10):
        sklutils.compare(sklgr, mylgr, datasets.load_digits(), score.accuracy, test_size=0.2, ohe=True, random_state=i)
    '''
