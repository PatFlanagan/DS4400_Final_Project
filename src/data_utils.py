import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    df = df[(df[["Squat1Kg", "Bench1Kg", "Deadlift1Kg"]] > 0).all(axis=1)]
    return df

def split_features_target(df: pd.DataFrame, features = FEATURES, target = TARGET):
    X = df[features].values
    y = df[target].values
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler

def prepare_data(path):
    df = load_data(path)
    X, y = split_features_target(df, FEATURES, TARGET)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_s, X_test_s, scaler = scale_data(X_train, X_test)
    return df, X_train_s, X_test_s, y_train, y_test