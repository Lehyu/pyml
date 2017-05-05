import numpy as np
forest_params_grid = {
    'n_estimators':[120, 300, 500, 800, 1200],
    'max_depth': [5, 8, 15, 25, 30, None],
    'min_samples_split':[2, 5, 10, 15, 100],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': [0.3,0.4,0.5,'log2', 'sqrt', 'auto']
}

ridge_params_grid = {
    "alpha":[0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "fit_intercept": [False, True],
    "normalize": [True, False]
}
ridge_grid_params_seq = [("alpha",),("fit_intercept",),("normalize",)]
xgb_params_grid = {
    'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
    'gamma': [0, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    'max_depth': [3, 5, 7, 9, 12, 15, 17, 25],
    'min_child_weight': [1, 3, 5, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9 ,1.0],
    'reg_alpha': [0, 0.1, 0.5, 1.0],
    'reg_lambda': [0.01, 0.04, 0.07, 0.1, 1.0],
    'colsample_bytree':[0.6, 0.7, 0.8, 1.0],
    'n_estimators': [120, 300, 500, 800, 1200, 2000, 3000]
}

gbdt_params_grid= {
    'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
    'n_estimators': [120, 300, 500, 800, 1200, 2000, 3000],
    'max_depth': [3, 5, 7, 9, 12, 15, 17, 25],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 5, 10],
    'max_features': ['log2', 'sqrt', 'auto'],
    'subsample': [0.6, 0.7, 0.8, 0.9 ,1.0]
}
gbdt_grid_search_params_seq = [
('max_depth', 'min_samples_split'), ('min_samples_leaf', ), ('max_features', 'subsample'), ('n_estimators', 'learning_rate')
]
xgb_grid_search_params_seq = [ ('max_depth', 'min_child_weight'), ('gamma',), ('subsample', 'colsample_bytree'),
                               ('reg_alpha', 'reg_lambda'), ('learning_rate', 'n_estimators') ]
forest_grid_search_params_seq = [('max_depth', 'min_samples_split'), ('min_samples_leaf', ), ('max_features',), ('n_estimators', )]
