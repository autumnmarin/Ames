import os
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import time
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from tqdm import tqdm  # ✅ Progress bar
import warnings
warnings.filterwarnings("ignore")
start_time = time.time()  # Start timer

# -------------------------------
# 1. Load Data (Including `training_extra.csv`)
# -------------------------------
seed = 87
script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, "raw", "train.csv")
train_extra_path = os.path.join(script_dir, "raw", "training_extra.csv")  # ✅ New file
test_path = os.path.join(script_dir, "raw", "test.csv")

df_train = pd.read_csv(train_path)
df_train_extra = pd.read_csv(train_extra_path)  # ✅ Load extra training data
df_test = pd.read_csv(test_path)

# 🔹 Concatenate `train.csv` and `training_extra.csv`
df_train = pd.concat([df_train, df_train_extra], ignore_index=True)  # ✅ Merge datasets

# 2. Apply Subsampling (Faster Training)
# -------------------------------
subsample_fraction = 0.01  # Train on 50% of the data (adjust as needed)
df_train = df_train.sample(frac=subsample_fraction, random_state=seed)


# Rename columns
df_train.rename(columns={"Weight Capacity (kg)": "Weight"}, inplace=True)
df_test.rename(columns={"Weight Capacity (kg)": "Weight"}, inplace=True)

# Remove ID columns
if "id" in df_train.columns:
    df_train.drop(columns=["id"], inplace=True)

if "id" in df_test.columns:
    test_ids = df_test["id"]
    df_test.drop(columns=["id"], inplace=True)
else:
    raise ValueError("The test.csv file must contain an 'id' column.")

# -------------------------------
# 2. Data Preprocessing
# -------------------------------
categorical_cols = ["Brand", "Material", "Size", "Laptop Compartment", "Waterproof", "Style", "Color"]
numerical_cols = ["Compartments", "Weight"] 

# Handle missing values
df_train[categorical_cols] = df_train[categorical_cols].fillna("Unknown")
df_test[categorical_cols] = df_test[categorical_cols].fillna("Unknown")

imputer = SimpleImputer(strategy="mean")
df_train[numerical_cols] = imputer.fit_transform(df_train[numerical_cols])
df_test[numerical_cols] = imputer.transform(df_test[numerical_cols])

# One-hot encoding (Ensures consistent feature columns)
df_train = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True)
df_test = pd.get_dummies(df_test, columns=categorical_cols, drop_first=True)

# 🔹 Ensure train & test have the same columns (align columns)
df_test = df_test.reindex(columns=df_train.columns.drop("Price"), fill_value=0)  # ✅ Ensures matching columns

# -------------------------------
# 3. Remove Outliers: Fashion Backpacks & IQR Filtering
# -------------------------------
# df_train = df_train[(df_train["Weight"] >= 0.75) | (df_train["Price"] > df_train["Price"].median())]

Q1 = df_train["Price"].quantile(0.25)
Q3 = df_train["Price"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR  # Adjust if needed (1.5 → 2.0 for stricter filtering)
upper_bound = Q3 + 1.5 * IQR
df_train = df_train[(df_train["Price"] >= lower_bound) & (df_train["Price"] <= upper_bound)]

# -------------------------------
# 4. Define Features & Target Variable
# -------------------------------
X_train = df_train.drop(columns=["Price"])  # ✅ Removes 'Price' to prevent feature mismatch
y_train = df_train["Price"]
X_test = df_test.copy()  # ✅ Ensures same columns

# Standardization
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# -------------------------------
# 5. Feature Engineering
# -------------------------------
skewed_features = ["Weight", "Compartments"]
for col in skewed_features:
    X_train[col] = X_train[col].clip(lower=0)  # ✅ Replace negative values with 0
    X_train[col].fillna(0, inplace=True)  # ✅ Replace NaNs with 0
    X_train[col + "_log"] = np.log1p(X_train[col])  # ✅ Apply log1p safely

    X_test[col] = X_test[col].clip(lower=0)  # ✅ Replace negative values with 0
    X_test[col].fillna(0, inplace=True)  # ✅ Replace NaNs with 0
    X_test[col + "_log"] = np.log1p(X_test[col])  # ✅ Apply log1p safely

X_train["Weight_Compartments"] = X_train["Weight"] * X_train["Compartments"]
X_test["Weight_Compartments"] = X_test["Weight"] * X_test["Compartments"]

# -------------------------------
# 6. Train Models with Hardcoded Best Parameters
# -------------------------------

## 🔹 XGBoost Best Parameters
xgb_best = xgb.XGBRegressor(
    objective="reg:squarederror",
    subsample=0.8,
    n_estimators=1000,
    max_depth=4,
    learning_rate=0.01,
    colsample_bytree=0.6,
    tree_method="gpu_hist",
    random_state=seed
)
xgb_best.fit(X_train, y_train)

## 🔹 LightGBM Best Parameters
use_gpu = lgb.__version__ >= "3.2.0"
lgb_best = lgb.LGBMRegressor(
    num_leaves=50,
    n_estimators=500,
    learning_rate=0.01,
    subsample=0.8,
    device="gpu" if use_gpu else "cpu",
    random_state=seed
)
lgb_best.fit(X_train, y_train)

## 🔹 CatBoost (Default Manually Set)
cat_model = CatBoostRegressor(iterations=1000, 
                              learning_rate=0.05, 
                              depth=6, random_state=seed, 
                              verbose=False)
cat_model.fit(X_train, y_train)

# -------------------------------
# 7. Use Stacking for Final Predictions
# -------------------------------
stacked_model = StackingRegressor(
    estimators=[
        ("xgb", xgb_best),
        ("lgb", lgb_best),
        ("cat", cat_model),
    ],
    final_estimator=Ridge()
)
stacked_model.fit(X_train, y_train)

# -------------------------------
# 8. Evaluate RMSE for Each Model
# -------------------------------
def evaluate_model(model, name):
    """Calculate and print RMSE for a given model."""
    y_pred = model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    print(f"📊 {name} RMSE: {rmse:.4f}")
    return rmse

xgb_rmse = evaluate_model(xgb_best, "XGBoost")
lgb_rmse = evaluate_model(lgb_best, "LightGBM")
cat_rmse = evaluate_model(cat_model, "CatBoost")
stacked_rmse = evaluate_model(stacked_model, "Stacked Model")

# Print all RMSEs
print("\n📈 RMSE Comparison:")
print(f"🔹 XGBoost RMSE: {xgb_rmse:.4f}")
print(f"🔹 LightGBM RMSE: {lgb_rmse:.4f}")
print(f"🔹 CatBoost RMSE: {cat_rmse:.4f}")
print(f"🔥 Stacked Model RMSE: {stacked_rmse:.4f}")

# -------------------------------
# 9. Make Predictions & Save Submission
# -------------------------------
y_pred = stacked_model.predict(X_test)

submission = pd.DataFrame({"id": test_ids, "Price": y_pred})
submission.to_csv("submission.csv", index=False)

print("\n✅ Predictions saved to 'submission.csv' with correct test IDs.")
# -------------------------------
# 14. Print Best Hyperparameters
# -------------------------------
print("\n🔍 Best Hyperparameters (Hardcoded):")
print("📌 XGBoost Best Params: {'subsample': 0.8, 'n_estimators': 1000, 'max_depth': 4, 'learning_rate': 0.01, 'colsample_bytree': 0.6}")
print("📌 LightGBM Best Params: {'num_leaves': 50, 'n_estimators': 500, 'learning_rate': 0.01}")
print("📌 CatBoost Params: Default (Manually Set)")
end_time = time.time()  # End timer
total_time = end_time - start_time
print(f"\n⏳ Total Runtime: {total_time:.2f} seconds")