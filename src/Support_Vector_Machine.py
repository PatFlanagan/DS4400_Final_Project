from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import pandas as pd
from pathlib import Path
from data_utils import (
    prepare_data,
    FEATURES
)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_PATH = DATA_DIR / "openpowerlifting.csv"

def create_elite_labels(y, percentile = .90):
    threshold  = np.percentile(y, percentile)
    return (y >= threshold).astype(int)

def main():
    df, X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)

    y_train_labels = create_elite_labels(y_train, 90)
    y_test_labels = create_elite_labels(y_test, 90)

    print("Class distribution (train):")
    print(pd.Series(y_train_labels).value_counts())

    print("\nTraining SVM Classifier...")

    svm_model = SVC(
        kernel="rbf",     # non-linear boundary
        C=1.0,            # regularization strength
        gamma="scale"    # default good starting point
    )

    svm_model.fit(X_train, y_train_labels)

    y_pred = svm_model.predict(X_test)

    print("\n=== SVM Results ===")
    print("Accuracy:", accuracy_score(y_test_labels, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test_labels, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_labels, y_pred))


if __name__ == "__main__":
    main()