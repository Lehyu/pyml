import numpy as np
from base import BaseEstimator
from metric.dist import Dist
from utils import ShuffleSpliter


class Cluster(object):
    def __init__(self, mu):
        self._mu = mu

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mu):
        self._mu = mu


class Kmeans(BaseEstimator):
    def __init__(self, k, tol=0.001):
        self.dist = Dist(criterion="Euclidean")
        self.k = k
        self.clusters = None
        self.tol = tol

    def fit(self, X, y=None):
        self._init_cluster(X)
        self._kmeans(X)

    def predict(self, X):
        return self._predict(X)

    def _kmeans(self, X):
        step = 1
        while step < 10000:
            C = [list() for i in range(self.k)]
            last_clusters = self.clusters.copy()
            # step 1. compute each samples cluster in current step
            for x in X:
                dists = []
                for i in range(self.k):
                    dists.append(self.dist.dist(x, self.clusters[i].mu))
                index = np.argmin(dists)
                C[index].append(x)

            # step2. update each cluster mean
            for i in range(self.k):
                Ci = np.asarray(C[i]).reshape((len(C[i]), -1))
                self.clusters[i].mu = np.mean(Ci, axis=0)

            # step3. compute the difference with last clusters
            diff = 0.0
            for i, cluster in enumerate(self.clusters):
                diff += self.dist.dist(cluster.mu, last_clusters[i].mu)
            diff /= self.k
            if diff <= self.tol:
                break
            step += 1

    def _init_cluster(self, X):
        spliter = ShuffleSpliter(len(X), test_size=len(X)-self.k)
        mu_slice, _ = spliter.shuffle()
        self.clusters = list()
        for ix in mu_slice:
            self.clusters.append(Cluster(X[ix]))

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
    mask = np.in1d(y, [0, 1])
    X_train, X_val, y_train, y_val = train_test_split(X[mask], y[mask], test_size=0.5, random_state=5)
    kmeans = Kmeans(2)
    kmeans.fit(X_train, y_train)
    y_pred = kmeans.predict(X_val)
    #y_pred = 1-y_pred
    print(y_pred)
    print(y_val)
    print("kmeans %.5f" % score.accuracy(y_pred, y_val))

