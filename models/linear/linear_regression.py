import numpy as np

from models.base_estimator import BaseEstimator
from optimizer.normal_equation import NormalEquation
from optimizer.sgd import Sgd
from utils.logger import logger


class LinearRegression(BaseEstimator):
    def __init__(self):
        self.optimizer = Sgd(learning_rate=0.001, max_iter=1000, batch_size=100, _lambda=0.05)
        # self.optimizer = NormalEquation()
        self.logger = logger("LinearRegression")
        self.params = dict()
        self.trained = False

    def fit(self, X, y=None):
        self.logger.check(X, y)
        n_samples, n_features = X.shape
        y = y.reshape((n_samples, -1))
        n_classes = y.shape[1]
        self.params["coef"] = np.random.uniform(-0.1, 0.1, (n_features, n_classes))
        self.params["bias"] = np.zeros(1)
        self.params = self.optimizer.solve(X, y, self.params)
        self.trained = True

    def feval(self, X, params):
        return np.dot(X, params['coef']) +params['bias']

    def predict(self, X):
        assert self.trained, "Please fit the model first!"
        return self.feval(X, self.params).reshape((X.shape[0]))


if __name__ == "__main__":
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression as SKLR
    from utils import sklutils
    from metric import metric as score

    mylr = LinearRegression()
    sklr = SKLR()
    sklutils.compare(sklr, mylr, load_diabetes(), score.metric, test_size=0.2)
