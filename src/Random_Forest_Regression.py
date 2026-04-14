from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import numpy as np
from visuals import plot_confusion_matrix, plot_feature_importance, plot_roc_curve

from data_utils import (
    prepare_data,
    FEATURES
)
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
DATA_PATH = DATA_DIR / "openpowerlifting.csv"

def create_labels(y, threshold):
    return (y >= threshold).astype(int)

def main():
    df, X_train, X_test, y_train, y_test = prepare_data(DATA_PATH)

    threshold = np.percentile(y_train, 85)
    y_train_labels = create_labels(y_train, threshold)
    y_test_labels = create_labels(y_test, threshold)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train_labels)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    print(classification_report(y_test_labels, y_pred))
    print(confusion_matrix(y_test_labels, y_pred))

    print("\n=== Random Forest Anomaly Detection ===")
    print(classification_report(y_test_labels, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_labels, y_pred))

    print("\n=== Feature Importance ===")
    for f, imp in zip(FEATURES, clf.feature_importances_):
        print(f"{f}: {imp:.4f}")
    
    print("\nGenerating plots...")
    
    plot_confusion_matrix(y_test_labels, y_pred, "rf_confusion_matrix.png")
    plot_feature_importance(FEATURES, clf.feature_importances_)
    plot_roc_curve(y_test_labels, y_prob, "rf_roc_curve.png")

if __name__ == "__main__":
    main()
