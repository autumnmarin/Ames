import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))

def load_data():
    """
    Load and preprocess training and testing data for two feature sets.
    """

    X_train_A = pd.read_csv('HousePricesTFDF/HousePricesTFDF/data/processed/X_train.csv')
    X_train_B = pd.read_csv('HousePricesTFDF/HousePricesTFDF/data/processed/X_train_eng.csv')
    X_test_A = pd.read_csv('HousePricesTFDF/HousePricesTFDF/data/processed/X_test.csv')
    X_test_B = pd.read_csv('HousePricesTFDF/HousePricesTFDF/data/processed/X_test_eng.csv')
    y_train_A = pd.read_csv('HousePricesTFDF/HousePricesTFDF/data/processed/y_train.csv').squeeze("columns")
    y_train_B = pd.read_csv('HousePricesTFDF/HousePricesTFDF/data/processed/y_train_eng.csv').squeeze("columns")    

    # Convert to Series if needed
    if isinstance(y_train_A, pd.DataFrame):
        y_train_A = y_train_A.squeeze()
    if isinstance(y_train_B, pd.DataFrame):
        y_train_B = y_train_B.squeeze()

    # **Check for mismatches before returning**
    print(f"Shapes: X_train_A: {X_train_A.shape}, y_train_A: {y_train_A.shape}")
    print(f"Shapes: X_train_B: {X_train_B.shape}, y_train_B: {y_train_B.shape}")
    assert X_train_A.shape[0] == y_train_A.shape[0], "Mismatch: X_train_A and y_train_A row counts differ!"
    assert X_train_B.shape[0] == y_train_B.shape[0], "Mismatch: X_train_B and y_train_B row counts differ!"

    return X_train_A, X_train_B, X_test_A, X_test_B, y_train_A, y_train_B

def preprocess_target(y):
    """Log-transform the target variable to handle skewness."""
    return np.log1p(y)

def split_data(X, y, test_size=0.2, random_state=87):
    """Split data into training and validation sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_model(model, X_train, y_train, RANDOM_STATE):
    """Evaluates model using cross-validation RMSE and R²."""
    from sklearn.model_selection import train_test_split, cross_val_score

    # Train-Test Split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Train Model
    model.fit(X_train_split, y_train_split)
    y_val_pred = model.predict(X_val_split)
    
    # RMSE Calculation
    scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    rmse_scores = np.sqrt(-scores)

    # R² Calculation
    r2 = r2_score(y_val_split, y_val_pred)

    return {
        "RMSE": rmse_scores.mean(),
        "RMSE_std": rmse_scores.std(),
        "R2": r2
    }

def make_predictions(model, X_train, y_train, X_test):
    """Train model and make predictions."""
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    return y_test_pred

def print_metrics(metrics):
    """Print formatted evaluation metrics."""
    for metric, value in metrics.items():
        print(f"{metric.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"{metric.replace('_', ' ').title()}: {value}")