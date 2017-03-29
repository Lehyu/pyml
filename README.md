Try to implement algorithms of machine learning which I known and used. These implementation refer to **sklearn**.

# Classifier

## SVM

## DecisionTreeClassifier

[2017.03.23] I haven't finished the pruning code yet.  
[2017.03.24] Finished the pruning code, it seems works quiet well on iris data set(sklearn)
[2017.03.27] Found a new issue of **min_samples_split** when I wrote the RandomForestClassifier code. I didn't handle this parameter for now.

## LogisticRegression
[2017.03.29] Finished. support multi-class, I haven't test binary classification yet, there are maybe some bug. 

## RandomForestClassifier

[2017.03.27] Finished, it seems work well on iris data set while on digits is quiet bad. 

# Regressor

## DecisionTreeRegressor
[2017.03.23] I haven't finished the pruning code yet.  
[2017.03.24] Finished the pruning code. but it seems quiet sucks on diabetes data set(sklearn). Maybe some part went wrong, still haven't no idea

## LinearRegression

[2017.03.28] Finished LR, it has comparable result on diabetes data set with LR in sklearn.

# Optimizer

## Stochastic Gradient Descent

[2017.03.28] Finished SGD for LR, there are maybe some bug when apply other learner. Remain to test. 

# Decomposition

## PCA
This is a simplified pca code. It will be more complex if we consider all the situation.