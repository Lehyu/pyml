import numpy as np

from base import BaseEstimator


class HMM(BaseEstimator):
    def __init__(self, n_components, n_outputs, startprob, transmat, outmat):
        self.n_components = n_components
        self.n_outputs = n_outputs
        self.startprob = startprob
        self.transmat = transmat
        self.outmat = outmat
        self.alpha = None
        self.beta = None

    def fit(self, X, y=None):
        pass

    def forward(self, X):
        T = len(X)
        alpha = np.ndarray((T, self.n_components))
        alpha[0] = self.startprob*self.outmat.T[X[1]]
        for t in range(T-2):
            for j in range(self.n_components):
                prev = np.sum(np.dot(alpha[t], self.transmat.T[j]))
                alpha[t+1][j] = prev*self.outmat[j][X[t+1]]
        return np.sum(alpha[T-1])

    def backward(self, X):
        T = len(X)
        beta = np.ndarray((T, self.n_components))
        beta[T-1] = np.array([1]*self.n_components)
        for t in range(T-2, -1, -1):
            for i in range(self.n_components):
                beta[t][i] = np.sum(self.transmat[i]*self.outmat.T[X[t+1]]*beta.T[t+1])
        return np.sum(self.startprob*self.outmat.T[X[1]]*beta[1])

    def viterbi(self, X):
        T = len(X)
        sigma = np.ndarray((T, self.n_components))
        thea = np.ndarray((T, self.n_components))
        sigma[0] = self.startprob*self.outmat.T[X[0]]
        thea[0] = np.array([0]*self.n_components)
        for t in range(1, T):
            for j in range(self.n_components):
                i = np.argmax(sigma[t-1]*self.transmat.T[j])
                thea[t][j] = i
                sigma[t][j] = sigma[t-1][i]*self.transmat[i][j]*self.outmat[j][X[t]]
        qt = np.argmax(sigma[T-1])
        prob = sigma[qt]
        Q = [qt]
        prev = qt
        for t in range(T-2, -1, -1):
            qt = thea[t+1][prev]
            prev = qt
            Q.append(qt)
        return prob, Q

    def forward_backward(self, X):
        pass


