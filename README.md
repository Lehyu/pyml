Try to implement algorithms of machine learning which I known and used. These implementation refer to **sklearn**.

# Classifier

## SVM

## DecisionTreeClassifier

[2017.03.23] ~~I haven't finished the pruning code yet.~~  
[2017.03.24] Finished the pruning code, it seems works quiet well on iris data set(sklearn)
[2017.03.27] Found a new issue of **min_samples_split** when I wrote the RandomForestClassifier code. I didn't handle this parameter for now.

## LogisticRegression
[2017.03.29] Finished. support multi-class and of course binary classification.
 
## RandomForestClassifier

[2017.03.27] Finished, it seems work well on iris data set while on digits is quiet bad. 

# Regressor

## DecisionTreeRegressor
[2017.03.23] ~~I haven't finished the pruning code yet.~~  
[2017.03.24] Finished the pruning code. but it seems quiet sucks on diabetes data set(sklearn). Maybe some part went wrong, still haven't no idea

## LinearRegression

[2017.03.28] Finished LR, it has comparable result on diabetes data set with LR in sklearn.

## Ridge

[2017.04.10] Base on the customs and usages, split Ridge from LinearRegressor

## Lasso

[2017.04.10] complete the forward selection optimizer. todo LARS and forward stagewise

# Optimizer

## Stochastic Gradient Descent

[2017.03.28] Finished SGD for LR, there are maybe some bug when apply other learner. Remain to test. 

# Model Selection

## split

[2017.04.05] split. p_train_test_split: split train and test dataset according to condition, input should be pandas.DataFrame.

[2017.04.08] split. CVSpliter: generate cross validation data set accoring cv(int).

## search 

[2017.04.07] GridSearchCV: use greedy strategy to search the params_grid; FullSearchCV: search params_grid on the whole parameter space.

[2017.04.08] SingleModelSearchCV: compare model with their default setting, or you can set the parameters beforehand.
 

# Decomposition

## PCA
This is a simplified pca code. It will be more complex if we consider all the situation.