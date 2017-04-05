import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from models.svm import SVC
from sklearn import svm
from utils import sklutils
from metric import score as score

if __name__ == '__main__':
    mysvc = SVC(C=1, kernel='rbf', max_iter=1, tol=0.001, sigma=0.5)
    sksvc = svm.SVC(C=1.0, kernel='rbf', gamma=0.5, tol=1e-3, max_iter=1)
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    mask = np.in1d(y, [0,1])
    X_train, X_val, y_train, y_val = train_test_split(X[mask], y[mask], test_size=0.5, random_state=5)
    mysvc.fit(X_train, y_train)
    sksvc.fit(X_train, y_train)
    print("mysvc %.5f" % score.accuracy(mysvc.predict(X_val), y_val))
    print("sksvc %.5f" % score.accuracy(sksvc.predict(X_val), y_val))

