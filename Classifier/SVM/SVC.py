import numpy as np
import random

from base.BaseEstimator import BaseEstimator
from utils.logger import logger


class SVC(BaseEstimator):
    def __init__(self, C=1, kernel='rbf', max_iter=10, tol=0.001, **kwargs):
        self.C = C
        self.kernel_type = kernel
        self.max_iter = max_iter
        self.tol = tol
        self.is_trained = False
        if self.kernel_type == 'rbf':
            if 'sigma' in kwargs:
                self.simga = kwargs['sigma']
            else:
                self.simga = 0.5
        elif self.kernel_type == 'poly':
            if 'degree' in kwargs:
                self.degree = kwargs['degree']
            else:
                self.degree = 3
        elif self.kernel_type == 'linear':
            if 'constant' in kwargs:
                self.constant = kwargs['constant']
            else:
                self.constant = 3
        self.logger = logger('SVC')

    def fit(self, X, y=None):
        if self.is_trained:
            return
        self.logger.check(X, y)
        n_size, n_features = X.shape
        self.n_features = n_features
        y = self._preprocess_y(y)
        self.alphas = np.zeros((n_size, 1), dtype=np.float64)
        self.ecache = np.zeros((n_size, 1), dtype=np.float64)
        self.b = np.zeros(1)
        K = np.zeros((n_size, n_size), dtype=np.float64)
        for ix in range(n_size):
            K[ix, :] = self._kernel_transform(X, X[ix])
        for ix in range(n_size):
            self._update_e(K, y, ix)
        self._platt_smo(K, y)
        indices = np.nonzero((self.alphas > 0) & (self.alphas < self.C))[0]
        self.sv = X[indices, :]
        self.y = y[indices]
        self.alphas = self.alphas[indices]
        self.is_trained = True

    def predict(self, X):
        assert self.is_trained, 'model hasn\'t been trained yet!'
        n_size, n_features = X.shape
        if n_features != self.n_features:
            raise IndexError("X index is (-1,%d) while it prefer (-1,%d)"%(n_features, self.n_features))
        pred = []
        for ix in range(n_size):
            p = np.dot(self._kernel_transform(self.sv, X[ix]), self.alphas*self.y)+self.b
            pred.append(np.sign(p))
        pred = np.asarray([1 if p == 1 else 0 for p in pred])
        pred = pred.reshape((-1,1))
        return pred

    def _kernel_transform(self, X, x):
        '''
        :param X: (n_samples, n_features)
        :param x: (1, n_features)
        :return:  (1, n_samples)
        '''
        if self.kernel_type == 'rbf':
            n_size = X.shape[0]
            k = np.zeros((1, n_size))
            for ix in range(n_size):
                delta = X[ix] - x
                k[:,ix] = np.dot(delta, delta)
            k = np.exp(k/(-2.0*self.simga**2))
            return k
        elif self.kernel_type == 'poly':
            return (np.dot(X, x)+1)**self.degree
        elif self.kernel_type == 'linear':
            return np.dot(X, x)+self.constant
        else:
            raise NameError('Check if the kernel splled right or not! '
                            'If it\'s right, then %s has been\'t supported yet!')

    def _preprocess_y(self, y):
        n_class = len(np.unique(y))
        if n_class != 2:
            raise ValueError('For now only support binary classification!')
        y = np.asarray([-1 if v == 0 else 1 for v in y])
        y = y.reshape((-1, 1))
        return y

    def _platt_smo(self, K, y):
        n_size = len(y)
        alpha_changed = 0
        entire_set = True
        iter_count = 0
        while (iter_count < self.max_iter) and (alpha_changed > 0 or entire_set):
            alpha_changed = 0
            if entire_set:
                for ix in range(n_size):
                    alpha_changed += self._examine_at(K, y, ix)
            else:
                non_bound_set = np.nonzero((self.alphas > 0) & (self.alphas < self.C))[0]

                for ix in non_bound_set:
                    alpha_changed += self._examine_at(K, y, ix)
            print('iteration %d examine at %s : %d alpha pair change!' % (
            iter_count,  'entire set' if entire_set else 'non bound set', alpha_changed))
            if entire_set:
                entire_set = False
            elif alpha_changed == 0:
                entire_set = True
            iter_count += 1


    def _examine_at(self, K, y, ix2):
        a2 = self.alphas[ix2]
        r2 = y[ix2] * self.ecache[ix2]
        '''
        KKT: 1) a = 0     and ty >= 1
             2) 0 < a < C and ty = 1
             3) a = C     and ty <= 1
        find alphas[ix] which violate KKT: r2 = ty-1, r2 < 0 ,r2 = 0 or r2 > 0
             1) r2 < 0 if a < C, violate
             2) r2 = 0 , none
             3) r2 > 0 if a > 0, violate
        self.tol is the tol that r2 can range
        '''
        if (r2 < -self.tol and a2 < self.C) or (r2 > self.tol and a2 > 0):
            non_bound_set = np.nonzero((self.alphas > 0) & (self.alphas < self.C))[0]
            e2 = self._calc_e(K, y, ix2)
            if len(non_bound_set) > 1:
                ix1, eix1 = self._choose_multiplier(K, y, ix2, e2, non_bound_set)
                if self._take_step(K, y, ix1, ix2, eix1, e2):
                    return 1
            for ix1 in non_bound_set:
                if ix1 == ix2:
                    continue
                eix1 = self._calc_e(K, y, ix1)
                if self._take_step(K, y, ix1, ix2, eix1, e2):
                    return 1
            for ix1 in range(len(y)):
                if ix1 == ix2:
                    continue
                eix1 = self._calc_e(K, y, ix1)
                if self._take_step(K, y, ix1, ix2, eix1, e2):
                    return 1
        return 0

    def _take_step(self, K, y, ix1, ix2, e1, e2):
        K11 = K[ix1, ix1]; K12 = K[ix1, ix2]; K22 = K[ix1, ix2]
        y1 = y[ix1]; y2 = y[ix2]; s = y1*y2
        a1 = np.copy(self.alphas[ix1])
        a2 = np.copy(self.alphas[ix2])
        # I don't have a clear idea why L H selected. todo
        if s == 1:
            L = max(0, a1+a2-self.C); H = min(self.C, a1+a2)
        else:
            L = max(0, a2-a1); H = min(self.C, self.C+a2-a1)
        if L == H: return False
        eta = K11 + K22 - 2*K12
        if eta <= 0: return False
        '''
        this is derive from the L(alpha), replace a1 and eliminate y1
        vi = f(xi) - sum(aj*yj*K(xi,xj)) i =1,2; j=3,....
        L(alpha) = L(a1, a2) = K11*a1**2/2+K22*a2**2/2+y1*y2*K12*a1*a2-(a1+a2)+y1*v1*a1+y2*v2*a2
        '''
        alpha2 = a2 + y2*(e1-e2)/eta
        alpha2 = self._clip_alpha(alpha2, L, H)
        alpha1 = a1 + s*(a2 - alpha2) # sum(ai*yi) = 0
        self.alphas[ix1] = alpha1
        self.alphas[ix2] = alpha2
        b1 = self.b - e1 - y1*K11*(alpha1 - a1) - y2*K12*(alpha2 - a2)
        b2 = self.b - e2 - y1*K12*(alpha1 - a1) - y2*K22*(alpha2 - a2)
        if 0 < alpha2 < self.C:
            self.b = b2
        elif 0 < alpha1 < self.C:
            self.b = b1
        else:
            self.b = (b1+b2)/2.0

        # it's not clear how to update e_cache uing only support vector or the whole dataset
        self._update_e(K, y, ix1)
        self._update_e(K, y, ix2)
        return True

    def _clip_alpha(self, alpha, L, H):
        if alpha < L:
            return L
        elif alpha > H:
            return H
        else:
            return alpha
    def _calc_e(self, K, y, ix):
        yi = y[ix]
        '''
        ei = fi-yi and fi = sum(aj*tj*K(xi,xj))+b
        '''
        fi = np.dot(K[ix], self.alphas*y)+self.b
        return float(fi-yi)

    def _update_e(self, K, y, ix):
        ei = self._calc_e(K, y, ix)
        self.ecache[ix] = ei

    def _choose_multiplier(self, K, y, ix2, e2, non_bound_set):
        max_step = 0
        ix1 = -1
        eix1 = 0
        if len(non_bound_set) > 1:
            for ix in non_bound_set:
                if ix == ix2:
                    continue
                eix = self._calc_e(K, y, ix)
                step = abs(e2 - eix)
                if step > max_step:
                    eix1 = eix
                    max_step = step
                    ix1 = ix
        else:
            ix1 = ix2
            while ix1 == ix2:
                ix1 = int(random.uniform(0, len(y)))
            eix1 = self._calc_e(K, y, ix1)
        return ix1, eix1

