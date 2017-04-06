import numpy as np
from sklearn.preprocessing import OneHotEncoder as encoder
from scipy import sparse


class OneHotEncoder(object):
    """
    categorical_features: "all" or array
    ------------------------------------------
    """
    def __init__(self,categorical_features="all"):
        self.categorical_features = categorical_features
        self.enc = encoder()

    def fit(self, X):
        n_features = X.shape[1]
        self._check(n_features)
        self.n_values = dict()
        self.cates_map = dict()
        for col in self.categorical_features:
            cates = self._get_cates(X[:, col])
            self.n_values[col] = len(cates)
            index = 1
            for val in cates:
                self.cates_map[col, val] = index
                index += 1
        X = self._transform(X)
        self.enc.fit(X[:, self.categorical_features])
        return self

    def transform(self, X):
        X = self._transform(X)
        n_features = X.shape[1]
        ind = np.arange(n_features)
        sel = np.zeros(n_features, dtype=bool)
        sel[np.asarray(self.categorical_features)] = True
        not_sel = np.logical_not(sel)

        X_selected = X[:, ind[sel]]
        X = X[:, ind[not_sel]]
        X_selected = self.enc.transform(X_selected).toarray()
        if sparse.issparse(X_selected) or sparse.issparse(X):
            return sparse.hstack((X, X_selected))
        else:
            return np.hstack((X,X_selected))

    def _transform(self, X):
        n_samples  = X.shape[0]
        for i in range(n_samples):
            for col in self.categorical_features:
                try:
                    X[i][col] = self.cates_map[col, X[i][col]]
                except:
                    raise ValueError('col {0} value {1} hasn\'t been seen yet\n'.format(col, X[i][col]))
        X = X.astype(np.float)
        return X

    def _get_cates(self, array):
        cates = set()
        for val in array:
            cates.add(val)
        return cates
    def _check(self, n_features):
        if self.categorical_features == "all":
            self.categorical_features = list(range(n_features))

        if not isinstance(self.categorical_features,list):
            raise TypeError('Parameter categorical_features must be '
                            '"all" or list not {0}'.format(type(self.categorical_features)))

        for col in self.categorical_features:
            if col > n_features:
                raise ValueError('feature col ')
