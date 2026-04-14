import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
import pandas as pd

from data_utils import (
    prepare_data,
    FEATURES
)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_PATH = DATA_DIR / "openpowerlifting.csv"

def add_intercept(X):
    return np.column_stack([np.ones(X.shape[0]), X])

def linear_regression_closed_form_with_ridge_penalty(X, y, alpha=1.0):
    Xb = add_intercept(X)
    n_features = Xb.shape[1]
    I = np.eye(n_features)
    I[0, 0] = 0
    theta = np.linalg.pinv(Xb.T @ Xb + alpha * I) @ (Xb.T @ y)
    return theta

def predict_linear(X, theta):
    Xb = add_intercept(X)
    return Xb @ theta

def main():
    print("Loading data...")
    df, X_train_s, X_test_s, y_train, y_test = prepare_data(DATA_PATH)

    print("\nTraining sklearn Ridge Regression...")
    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)

    y_train_pred = model.predict(X_train_s)
    y_test_pred = model.predict(X_test_s)

    print("\n=== Ridge Sklearn Results ===")
    print("Train MSE:", mean_squared_error(y_train, y_train_pred))
    print("Test MSE:", mean_squared_error(y_test, y_test_pred))
    print("Train R2:", r2_score(y_train, y_train_pred))
    print("Test R2:", r2_score(y_test, y_test_pred))

    print("\nTraining closed-form Ridge Regression...")
    theta = linear_regression_closed_form_with_ridge_penalty(
        X_train_s, y_train, alpha=1.0
    )

    y_train_pred_cf = predict_linear(X_train_s, theta)
    y_test_pred_cf = predict_linear(X_test_s, theta)

    print("\n=== Closed-Form Ridge Results ===")
    print("Train MSE:", mean_squared_error(y_train, y_train_pred_cf))
    print("Test MSE:", mean_squared_error(y_test, y_test_pred_cf))
    print("Train R2:", r2_score(y_train, y_train_pred_cf))
    print("Test R2:", r2_score(y_test, y_test_pred_cf))

    print("\n=== Feature Coefficients ===")
    for f, c in zip(FEATURES, model.coef_):
        print(f"{f}: {c:.4f}")

    # anomaly detection (based on sklearn model)
    residuals = y_test - y_test_pred
    idx = np.argsort(np.abs(residuals))[::-1][:10]

    print("\n=== Top 10 Anomalies ===")
    print(df.iloc[idx])

if __name__ == "__main__":
    main()