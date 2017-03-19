from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

from Classifier.SVM.SVC import SVC

if __name__ == '__main__':
    iris = load_iris()
    X, y = iris.data, iris.target
    y = y.reshape((-1, 1))
    indices = [i for i in range(len(y)) if y[i] == 0 or y[i] == 1]

    y = y[indices,:]
    X = X[indices,:]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    kwargs = {'sigma': 0.5}
    svm = SVC(C=1, kernel='linear',constant=3)
    svm.fit(X_train, y_train)
    pred = svm.predict(X_val)
    print('error ',np.sum(abs(pred-y_val))/len(pred))
