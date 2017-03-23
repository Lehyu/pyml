from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

from base.CART import DecisionTreeClassifier #, DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor

def tree_test_sklearn_data(data, model='DecisionTreeClassifier', criterion='Gini'):
    X, y = data.data, data.target
    X_train, X_val, y_train, y_val = train_test_split(X, y)
    tree = eval(model)(criterion=criterion)
    tree.fit(X_train, y_train)

    pred = tree.predict(X_val)
    print([v for v in pred])
    print([v for v in y_val])
    #print(tree.score(pred, y_val))
    print(sklearn_mse(pred, y_val))
def sklearn_mse(predict, y_val):
    return np.sqrt(np.sum((predict-y_val)**2))
if __name__ == '__main__':
    # test SVC
    ''''''
    #tree_test_sklearn_data(datasets.load_iris(), model='DecisionTreeClassifier')
    tree_test_sklearn_data(datasets.load_diabetes(), model='DecisionTreeRegressor', criterion='mse')
