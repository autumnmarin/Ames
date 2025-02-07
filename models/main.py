import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from util import load_data, preprocess_target, make_predictions, print_metrics, evaluate_model

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

RANDOM_STATE = 87

# Load data for two feature sets
X_train_A, X_train_B, X_test_A, X_test_B, y_train_A, y_train_B = load_data()

# Preprocess targets (log-transform to reduce skewness)
y_train_transformed_A = preprocess_target(y_train_A).squeeze()
y_train_transformed_B = preprocess_target(y_train_B).squeeze()


# Define candidate models
models = {
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "Linear Regression": LinearRegression(),
    "Gradient Boosting": HistGradientBoostingRegressor(
        learning_rate=0.1, max_iter=150, max_depth=3, min_samples_leaf=5, random_state=RANDOM_STATE
    ),
    "XGBoost": XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=3, colsample_bytree= 0.6, gamma= 0, subsample= 0.6, random_state=RANDOM_STATE, verbosity=0
    ),
    "CatBoost": CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=6, random_seed=RANDOM_STATE, verbose=10
    ),
}

performance_results = {}
best_model_info = None  # Tuple: (model_name, model, feature_set, X_test)
best_rmse = float("inf")

# Evaluate each model on both feature sets
for feature_set, (X_train, X_test, y_transformed) in zip(
    ["Raw Features", "Engineered Features"],
    [(X_train_A, X_test_A, y_train_transformed_A), (X_train_B, X_test_B, y_train_transformed_B)]
):
    print(f"\nEvaluating models using {feature_set}...\n")
    model_performance = {}
    for model_name, model in models.items():
        print(f"Evaluating: {model_name} on {feature_set}")
        metrics = evaluate_model(model, X_train, np.expm1(y_transformed), RANDOM_STATE)
        if "RMSE" in metrics:
            model_performance[model_name] = metrics
            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                best_model_info = (model_name, model, feature_set, X_test)
        else:
            print(f"Warning: 'RMSE' not found in metrics for {model_name}. Skipping.")
        print_metrics({k: round(v, 2) if isinstance(v, float) else v for k, v in metrics.items()})
        print("-" * 40)
    performance_results[feature_set] = model_performance

if best_model_info is None:
    raise ValueError("No valid model was found during evaluation.")

best_model_name, best_model, best_feature_set, best_X_test = best_model_info
print(f"\nBest Model: {best_model_name} ({best_feature_set}) with RMSE: {best_rmse:.2f}")

# Refit the best model on the full training data for that feature set.
if best_feature_set == "Raw Features":
    # Use the appropriate raw feature training data and target
    best_model.fit(X_train_A, np.expm1(y_train_transformed_A))
elif best_feature_set == "Engineered Features":
    best_model.fit(X_train_B, np.expm1(y_train_transformed_B))

# Now generate predictions
predictions = best_model.predict(best_X_test)

# Create submission DataFrame with Id starting at 1461
submission = pd.DataFrame({
    "Id": range(1461, 1461 + len(predictions)),
    "SalePrice": predictions
})
submission.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
print("File location:", os.path.abspath("submission.csv"))

# Plot RMSE and R² scores for each model and feature set
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
colors = {"Raw Features": "green", "Engineered Features": "blue"}

for feature_set, perf in performance_results.items():
    model_names = list(perf.keys())
    rmse_values = [perf[model]["RMSE"] for model in model_names]
    r2_values = [perf[model]["R2"] for model in model_names]
    color = colors[feature_set]
    axes[0].plot(model_names, rmse_values, marker="o", linestyle="-", color=color, label=feature_set)
    axes[1].plot(model_names, r2_values, marker="s", linestyle="--", color=color, label=feature_set)

axes[0].set_title("Model Performance: RMSE Comparison")
axes[0].set_xlabel("Models")
axes[0].set_ylabel("RMSE (Lower is Better)")
axes[0].legend()
axes[0].grid(True)

axes[1].set_title("Model Performance: R² Score Comparison")
axes[1].set_xlabel("Models")
axes[1].set_ylabel("R² Score (Higher is Better)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("feature_set_comparison.png")
print("Feature set comparison chart saved as 'feature_set_comparison.png'")
plt.show()
