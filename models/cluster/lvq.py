import random

import numpy as np
from base import BaseEstimator
from metric.dist import Dist
from models.cluster.kmeans import Cluster


class LVQ(BaseEstimator):
    def __init__(self, reg_lambda=0.1, tol=1e-5):
        self.dist = Dist(criterion="Euclidean")
        self.clusters = None
        self.tol = tol
        self.reg_lambda = reg_lambda

    def fit(self, X, y=None):
        self._init_cluster(X, y)
        self._lvq(X, y)

    def predict(self, X):
        return self._predict(X)

    def _lvq(self, X, y):
        step = 1
        n_samples = X.shape[0]
        while step < 10000:
            index = random.sample(range(n_samples), 1)[0]
            ## compute distance of x_i and cluster_i
            dists = []
            for i in range(self.k):
                dists.append(self.dist.dist(X[index], self.clusters[i].mu))
            ## Find the minimum dist
            index = np.argmin(dists)
            if y[index] == index:
                p_ = self.clusters[index].mu + self.reg_lambda*(X[index]-self.clusters[index].mu)
            else:
                p_ = self.clusters[index].mu - self.reg_lambda * (X[index] - self.clusters[index].mu)
            self.clusters[index].mu = p_
            step += 1

    def _init_cluster(self, X, y):
        t = sorted(np.unique(y))
        self.k = len(t)
        self.clusters = []
        for ti in t:
            index = random.sample(list(np.nonzero(y==ti)[0]), 1)[0]
            self.clusters.append(Cluster(X[index]))


    def _predict(self, X):
        preds = []
        for x in X:
            dists = []
            for cluster in self.clusters:
                dists.append(self.dist.dist(x, cluster.mu))
            pred = np.argmin(dists)
            preds.append(pred)
        preds = np.asarray(preds).reshape((len(X),))
        return preds


if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from metric import score as score
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5, random_state=5)
    lvq = LVQ(0.0001)
    lvq.fit(X_train, y_train)

    y_pred = lvq.predict(X_val)
    #y_pred = 1-y_pred
    print(y_pred)
    print(y_val)
    print("kmeans %.5f" % score.accuracy(y_pred, y_val))