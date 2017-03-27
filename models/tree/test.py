import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as SKLRF

from models.tree import cart
from sklearn import tree
import sklearn
from models.tree.RandomForest import RandomForestClassifier
from utils import sklutils
from metric import metric as score

if __name__ == '__main__':
    #'''
    # compare decision tree classifier
    mytree = cart.DecisionTreeClassifier(criterion='gini')
    sktree = tree.DecisionTreeClassifier(criterion='gini')
    sklutils.compare(sktree, mytree, datasets.load_digits(),score.accuracy, test_size=0.5, random_state=12)
    #'''

    '''
    # compare decision tree regressor
    mytree = cart.DecisionTreeRegressor(criterion='mse', max_depth=None, min_samples_leaf=2)
    sktree = tree.DecisionTreeRegressor(criterion='mse', max_depth=4, min_samples_leaf=2)
    sklutils.compare(sktree, mytree, datasets.load_diabetes(), score.metric)
    #'''


    #'''
    for i in range(10):
        RF = RandomForestClassifier(max_features="log2", min_samples_splits=10)
        skrf = SKLRF(max_features="log2")
        print('random state %d'%i)
        sklutils.compare(skrf, RF, datasets.load_digits(), score.accuracy, test_size=0.2, random_state=i)
    #'''


