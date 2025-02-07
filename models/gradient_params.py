import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

def tune_gradient_boosting(X_train, y_train, random_state=87):
    """
    Perform hyperparameter tuning for Gradient Boosting Regressor.

    Parameters:
    - X_train: Training features.
    - y_train: Training target.
    - random_state: Random seed for reproducibility.

    Returns:
    - best_model: GradientBoostingRegressor with the best found parameters.
    - best_params: Dictionary of the best parameters.
    """
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    }

    # Initialize the model
    gbm = GradientBoostingRegressor(random_state=87)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=gbm,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Retrieve the best parameters and model
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    return best_model, best_params
