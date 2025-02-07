# Xg boost param

####################################

import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import numpy as np

# Initialize an XGBoost regressor
xgb_reg = xgb.XGBRegressor(
    objective='reg:squarederror',  # use squared error for regression
    random_state=RANDOM_STATE
)

# Define a parameter grid to search
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# Set up GridSearchCV using negative mean squared error as the scoring metric
grid_search = GridSearchCV(
    estimator=xgb_reg,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',  # grid search maximizes score, so neg MSE is used
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit the grid search on your training data
grid_search.fit(X_train_A, y_train_transformed_A)

# Extract the best parameters and corresponding RMSE
best_params = grid_search.best_params_
best_rmse = np.sqrt(-grid_search.best_score_)

print("Best parameters:", best_params)
print("Best RMSE:", best_rmse)
###############################