import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
import os

OUTPUT_DIR = "outputs/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_pred_vs_actual(y_test, y_pred):
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             linestyle='--')

    plt.xlabel("Actual TotalKg")
    plt.ylabel("Predicted TotalKg")
    plt.title("Predicted vs Actual Performance")

    path = os.path.join(OUTPUT_DIR, "pred_vs_actual.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_residuals(y_test, y_pred):
    residuals = y_test - y_pred

    plt.figure(figsize=(6,5))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, linestyle='--')

    plt.xlabel("Predicted TotalKg")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")

    path = os.path.join(OUTPUT_DIR, "residuals.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_feature_importance(features, importances):
    plt.figure(figsize=(8,5))
    plt.barh(features, importances)
    plt.gca().invert_yaxis()

    plt.xlabel("Importance")
    plt.title("Feature Importance")

    path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(path, dpi=300)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    plt.figure()
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title("Confusion Matrix")

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()


def plot_roc_curve(y_true, y_prob, filename="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()