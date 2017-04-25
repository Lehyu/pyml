from collections import defaultdict

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from base import BaseEstimator
from utils import logger


class AdaBoost(BaseEstimator):
    def fit(self, X, y=None):
        pass


class AdaBoostClassifier(AdaBoost):
    def __init__(self, n_estimators=10, base_estimator=DecisionTreeClassifier, epsilon=1e-10):
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.estimators = [self.base_estimator() for i in range(self.n_estimators)]
        self.alphas = []
        self.logger = logger("AdaboostClassifier")
        self.epsilon = epsilon

    def fit(self, X, y=None):
        self.logger.check(X, y)
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        self._fit(X, y)

    def predict(self, X):
        assert X.shape[1] == self.n_features, "Train features %d while input %d" % (self.n_features, X.shape[1])
        n_samples = X.shape[0]
        pred = [defaultdict(float) for i in range(n_samples)]
        for t in range(len(self.alphas)):
            py = self.estimators[t].predict(X)
            for i, v in enumerate(py):
                pred[i][v] += self.alphas[t]
        for i in range(n_samples):
            max_prob = None
            label = None
            for k, v in pred[i].items():
                if max_prob is None or max_prob < v:
                    max_prob = v
                    label = k
            pred[i] = label
        return np.asarray(pred)

    def predict_proba(self, X):
        pass

    def _fit(self, X, y):
        n_samples = X.shape[0]
        weights = np.asarray([1.0 / n_samples] * n_samples)
        alphas_ = []
        for t in range(self.n_estimators):
            ## sample data set D_t
            if len(np.unique(weights)) == 1:
                sample_ix = list(range(n_samples))
            else:
                sample_ix = np.random.choice(range(n_samples), n_samples, True, weights)

            ## train model
            self.estimators[t].fit(X[sample_ix], y[sample_ix])
            py = self.estimators[t].predict(X)
            hits = np.asarray([1 if t != p else 0 for t, p in zip(y, py)])
            error = np.dot(hits, weights)
            if error > 0.5:
                break

            ## update model's weight
            #alphas_.append(np.log((1 - error) / (error)) / 2.0)
            ## adding epsilon to keep stability in case of strong classifier
            alphas_.append(np.log((1 - error+self.epsilon) / (error+self.epsilon)) / 2.0)

            ## update samples' weight
            weights = self._update_weights(weights, alphas_[t], y, py)
        self.alphas = np.asarray(alphas_)
        self.alphas/=np.sum(self.alphas)

    def _update_weights(self, weights, alpha, y, py):
        hits = np.asarray([1 if t != p else -1 for t, p in zip(y, py)])
        factors = np.asarray([alpha] * len(weights))
        factors = np.exp(hits * factors)
        weights *= factors
        z = np.sum(weights)
        weights /= z
        return weights


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from metric import score as score

    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=5)
    adabc = AdaBoostClassifier()
    adabc.fit(X_train, y_train)

    y_pred = adabc.predict(X_val)
    # y_pred = 1-y_pred
    print(y_pred)
    print(y_val)
    print("kmeans %.5f" % score.accuracy(y_pred, y_val))
