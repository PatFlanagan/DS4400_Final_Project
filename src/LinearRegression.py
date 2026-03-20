import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


from pathlib import Path
import pandas as pd

#repo root = parent of src/
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_PATH = DATA_DIR / "openpowerlifting.csv"

FEATURES = [
    "Squat1Kg",
    "Bench1Kg",
    "Deadlift1Kg",
    "BodyweightKg",
    "Age",
]
TARGET = "TotalKg"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run data_loader.py first."
        )

    cols = FEATURES + [TARGET]
    df = pd.read_csv(path, usecols=cols)
    df = df.dropna()
    return df

def add_intercept(X):
    return np.column_stack([np.ones(X.shape[0]), X])


def linear_regression_closed_form(X, y):
    Xb = add_intercept(X)
    theta = np.linalg.pinv(Xb.T @ Xb) @ (Xb.T @ y)
    return theta


def predict_linear(X, theta):
    Xb = add_intercept(X)
    return Xb @ theta


def load_data(path):
    # Only load columns you need
    cols = FEATURES + [TARGET]

    df = pd.read_csv(path, usecols=cols)

    # Drop missing values
    df = df.dropna()

    return df


def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Dataset size:", df.shape)

    #extracts X and y
    X = df[FEATURES].values
    y = df[TARGET].values

    #train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)


    print("\nTraining sklearn Linear Regression...")
    model = LinearRegression()
    model.fit(X_train_s, y_train)

    y_train_pred = model.predict(X_train_s)
    y_test_pred = model.predict(X_test_s)

    print("\n=== Sklearn Results ===")
    print("Train MSE:", mean_squared_error(y_train, y_train_pred))
    print("Test MSE:", mean_squared_error(y_test, y_test_pred))
    print("Train R2:", r2_score(y_train, y_train_pred))
    print("Test R2:", r2_score(y_test, y_test_pred))


    print("\nTraining closed-form Linear Regression...")
    theta = linear_regression_closed_form(X_train_s, y_train)

    y_train_pred_cf = predict_linear(X_train_s, theta)
    y_test_pred_cf = predict_linear(X_test_s, theta)

    print("\n=== Closed-Form Results ===")
    print("Train MSE:", mean_squared_error(y_train, y_train_pred_cf))
    print("Test MSE:", mean_squared_error(y_test, y_test_pred_cf))
    print("Train R2:", r2_score(y_train, y_train_pred_cf))
    print("Test R2:", r2_score(y_test, y_test_pred_cf))

    print("\n=== Feature Coefficients ===")
    for f, c in zip(FEATURES, model.coef_):
        print(f"{f}: {c:.4f}")

    residuals = y_test - y_test_pred

    idx = np.argsort(np.abs(residuals))[::-1][:10]

    print("\n=== Top 10 Anomalies ===")
    print(df.iloc[idx])


if __name__ == "__main__":
    main()