import numpy as np
from scipy import stats

class Imputer(object):
    '''
    missing_values: integer, str or "nan" or list
        default: 'nan'
    usc_cols: list or "all", default("all")
        which cols to be imputer
    strategy: str , default("most_frequent")
     If "mean" or "median", the replace missing values type must be integer
     If "most_frequent"
    '''
    def __init__(self, missing_values="nan", use_cols="all",strategy="most_frequent"):
        self.missing_values = missing_values
        self.use_cols = use_cols
        self.strategy = strategy

    def fit(self, X, y=None):
        self._check(X.shape[1])
        self.imputer_values = []
        for i in range(len(self.use_cols)):
            if self.strategy[i] == "most_frequent":
                val = self._most_frequent(X[:, self.use_cols[i]], self.missing_values[i])
            if self.strategy[i] == "mean":
                val = self._mean(X[:, self.use_cols[i]], self.use_cols[i], self.missing_values[i])
            if self.strategy[i] == "median":
                val = self._median(X[:, self.use_cols[i]], self.use_cols[i], self.missing_values[i])
            self.imputer_values.append(val)
        return self


    def transform(self, X):
        for i in range(len(X)):
            for j in range(len(self.use_cols)):
                if str(X[i][self.use_cols[j]]) == self.missing_values[j]:
                    X[i][self.use_cols[j]] = self.imputer_values[j]
        return X

    def _check(self, length):
        _strategies = ["mean", "median", "most_frequent"]
        if self.use_cols == "all":
            self.use_cols = list(range(length))
            print(self.use_cols)

        if type(self.missing_values) == str:
            self.missing_values = [self.missing_values for i in range(len(self.use_cols))]

        if type(self.strategy) == str:
            self.strategy = [self.strategy for i in range(len(self.use_cols))]

        if type(self.use_cols) != list:
            raise TypeError('Parameter use_cols must be str or list not {0}'.format(type(self.use_cols)))

        if type(self.missing_values) != list:
            raise TypeError('Parameter missing_values must be str or list not {0}'.format(type(self.missing_values)))

        if type(self.strategy) != list:
            raise TypeError('Parameter strategy must be str or list not {0}'.format(type(self.use_cols)))

        if len(self.missing_values) != len(self.use_cols):
            raise ValueError('len(use_cols) != len(missing_values)')

        if len(self.strategy) != len(self.use_cols):
            raise ValueError('len(use_cols) != len(strategy)')

        for val in self.strategy:
            if val not in _strategies:
                raise ValueError('Parameter strategy must be in {0} not {1}'.format(_strategies, val))

    def _most_frequent(self, array, missing_value):
        array = [x for x in array if str(x) != missing_value]
        try:
            array = array.astype(np.float)
            return stats.mode(array)[0][0]
        except:
            hist = dict()
            for val in array:
                hist[val] = hist[val]+1 if val in hist else 1
            most = 0
            ans = None
            for key, val in hist.items():
                if val > most:
                    most = val
                    ans = key
            return ans

    def _mean(self, array, col, missing_value):
        array = np.asarray([x for x in array if str(x) != missing_value])
        try:
            array = array.astype(np.float)
            return np.mean(array)
        except:
            raise TypeError("col {0}:Data can't be converted to float".format(col))

    def _median(self, array, col, missing_value):
        array = np.asarray([x for x in array if str(x) != missing_value])
        try:
            array = array.astype(np.float)
            return np.median(array)
        except:
            raise TypeError("col {0}:Data can't be converted to float".format(col))