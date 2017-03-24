import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from models.tree import cart
from sklearn import tree
from utils import sklutils
from metric import metric as score

if __name__ == '__main__':
    #'''
    # compare decision tree classifier
    mytree = cart.DecisionTreeClassifier(criterion='gini')
    sktree = tree.DecisionTreeClassifier(criterion='gini')
    sklutils.compare(sktree, mytree, datasets.load_iris(),score.accuracy, test_size=0.5, random_state=12)
    #'''

    # compare decision tree regressor
    mytree = cart.DecisionTreeRegressor(criterion='mse', max_depth=None, min_samples_leaf=2)
    sktree = tree.DecisionTreeRegressor(criterion='mse', max_depth=4, min_samples_leaf=2)
    sklutils.compare(sktree, mytree, datasets.load_diabetes(), score.metric)



