import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from models.tree import cart
from sklearn import tree
from utils import sklutils
from metric import metric as score

if __name__ == '__main__':
    '''
    # compare decision tree classifier
    mytree = CART.DecisionTreeClassifier(criterion='Gini')
    sktree = tree.DecisionTreeClassifier(criterion='gini')
    sklutils.compare(sktree, mytree, datasets.load_iris(),score.accuracy, test_size=0.5, random_state=10)
    '''

    # compare decision tree regressor
    mytree = cart.DecisionTreeRegressor(criterion='MSE')
    sktree = tree.DecisionTreeRegressor(criterion='mse')
    sklutils.compare(sktree, mytree, datasets.load_diabetes(), score.mse)



