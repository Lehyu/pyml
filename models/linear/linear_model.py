import numpy as np

from models.base_estimator import BaseEstimator
from optimizer.normal_equation import NormalEquation
from optimizer.sgd import Sgd
from utils import nutils
from utils.logger import logger

class LinearModel(BaseEstimator):

    def fit(self, X, y=None):
        self.logger.check(X, y)
        n_samples, n_features = X.shape
        y = y.reshape((n_samples, -1))
        n_classes = y.shape[1]
        self.params["coef"] = np.random.uniform(-0.1, 0.1, (n_features, n_classes))
        self.params["bias"] = np.zeros(1)
        self.params = self.optimizer.solve(X, y, self.params)
        self.trained = True


class LinearRegression(LinearModel):
    def __init__(self, learning_rate=1e-1, eps=1e-5, max_iter=1000, batch_size=10, loss="SumOfSquares", decay='step', _lambda=0.1):
        self.optimizer = Sgd(learning_rate=learning_rate, eps=eps, max_iter=max_iter, batch_size=batch_size, loss=loss,
                             decay=decay, _lambda=_lambda)
        # self.optimizer = NormalEquation()
        self.logger = logger("LinearRegression")
        self.params = dict()
        self.trained = False

    def predict(self, X):
        assert self.trained, "Please fit the model first!"
        return (np.dot(X, self.params['coef']) +self.params['bias']).reshape((X.shape[0]))

class LogisticRegression(LinearModel):
    def __init__(self, learning_rate=1e-3, eps=1e-5, max_iter=1000, batch_size=10, loss="LossWithSoftmax", decay='step', _lambda=0.1):
        self.loss = loss
        self.optimizer = Sgd(learning_rate=learning_rate, eps=eps, max_iter=max_iter, batch_size=batch_size, loss=loss, decay=decay, _lambda=_lambda)
        self.logger = logger("LogisticRegression")
        self.params = dict()
        self.trained = False

    def predict(self, X):
        if self.loss == "LossWithLogits":
            return nutils.sigmoid(np.dot(X, self.params['coef']) +self.params['bias']).reshape((X.shape[0]))
        elif self.loss == "LossWithSoftmax":
            pred = nutils.sofmax(np.dot(X, self.params['coef']) + self.params['bias'])
            return pred.argmax(axis=1)


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.linear_model import LinearRegression as SKLR
    from sklearn.linear_model import LogisticRegression as SKLGR
    from utils import sklutils
    from metric import metric as score
    '''
    mylr = LinearRegression()
    sklr = SKLR()
    sklutils.compare(sklr, mylr, datasets.load_diabetes(), score.metric, test_size=0.2)
    #'''

    mylgr = LogisticRegression()
    sklgr = SKLGR(multi_class='multinomial', solver="lbfgs")
    for i in range(10):
        sklutils.compare(sklgr, mylgr, datasets.load_digits(), score.accuracy, test_size=0.2, ohe=True, random_state=i)