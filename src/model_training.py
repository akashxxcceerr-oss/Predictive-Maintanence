"""
model_training.py
─────────────────
Trains, evaluates, and saves multiple scikit-learn classifiers for
multi-class failure-type prediction.

Failure label mapping (set in data_preprocessing.py):
  0 – No Failure
  1 – Heat Dissipation Failure
  2 – Overstrain Failure
  3 – Power Failure
  4 – Tool Wear Failure

Models trained:
  • Random Forest           (primary / best expected)
  • Gradient Boosting       (XGBoost-style via sklearn)
  • Logistic Regression     (baseline)
  • K-Nearest Neighbours    (for comparison)
"""

import os
import logging
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

LABEL_NAMES = [
    "No Failure",
    "Heat Dissipation",
    "Overstrain",
    "Power Failure",
    "Tool Wear",
    "Random Failure",
]

MODEL_DIR  = os.getenv("MODEL_DIR",  "models/")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs/")
RS         = int(os.getenv("RANDOM_STATE", "42"))


# ─────────────────────────────────────────────────────────────────────────────
def _get_models() -> dict:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RS,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=RS,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RS,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7,
            weights="distance",
            n_jobs=-1,
        ),
    }


def train_and_evaluate(
    X_train: np.ndarray,
    X_test:  np.ndarray,
    y_train: np.ndarray,
    y_test:  np.ndarray,
    feature_names: list,
) -> dict:
    """
    Train all models, evaluate on the test set, and persist the best one.

    Returns
    -------
    results : dict   – metrics for every trained model
    """
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    models  = _get_models()
    results = {}

    for name, clf in models.items():
        logger.info("Training %s ...", name)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average="macro",    zero_division=0)
        f1_wtd = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # 5-fold CV accuracy on train set
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5,
                                    scoring="accuracy", n_jobs=-1)

        present_classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
        report = classification_report(
            y_test, y_pred,
            labels=present_classes,
            target_names=[LABEL_NAMES[i] for i in present_classes],
            zero_division=0,
        )

        logger.info(
            "%s  |  Acc=%.4f  F1-macro=%.4f  CV=%.4f±%.4f",
            name, acc, f1_mac, cv_scores.mean(), cv_scores.std(),
        )
        logger.info("\n%s", report)

        results[name] = {
            "accuracy":      round(acc,    4),
            "f1_macro":      round(f1_mac, 4),
            "f1_weighted":   round(f1_wtd, 4),
            "cv_mean":       round(cv_scores.mean(), 4),
            "cv_std":        round(cv_scores.std(),  4),
            "clf":           clf,
            "y_pred":        y_pred,
            "report":        report,
        }

        # Save every model
        model_path = os.path.join(MODEL_DIR, f"{name}.joblib")
        joblib.dump(clf, model_path)
        logger.info("Saved model -> %s", model_path)

    # ── Pick the best model by weighted F1 ───────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["f1_weighted"])
    logger.info("Best model: %s  (F1-weighted=%.4f)",
                best_name, results[best_name]["f1_weighted"])

    best_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(results[best_name]["clf"], best_path)
    logger.info("Best model saved -> %s", best_path)

    # ── Save metrics summary ──────────────────────────────────────────────────
    summary = {
        k: {m: v for m, v in v_dict.items() if m != "clf" and m != "y_pred" and m != "report"}
        for k, v_dict in results.items()
    }
    summary["best_model"] = best_name
    metrics_path = os.path.join(OUTPUT_DIR, "metrics_summary.json")
    with open(metrics_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Metrics saved -> %s", metrics_path)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_confusion_matrices(results, y_test, output_dir=OUTPUT_DIR)
    _plot_model_comparison(results, output_dir=OUTPUT_DIR)

    if hasattr(results[best_name]["clf"], "feature_importances_"):
        _plot_feature_importance(
            results[best_name]["clf"],
            feature_names,
            model_name=best_name,
            output_dir=OUTPUT_DIR,
        )

    return results


# ── Plot helpers ──────────────────────────────────────────────────────────────
def _plot_confusion_matrices(results: dict, y_test: np.ndarray,
                              output_dir: str) -> None:
    present_classes = sorted(np.unique(y_test))
    labels = [LABEL_NAMES[i] for i in present_classes]

    ncols = len(results)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    if ncols == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"], labels=present_classes)
        sns_like_heatmap(ax, cm, labels)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle("Confusion Matrices", fontsize=13)
    fig.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved confusion_matrices.png")


def sns_like_heatmap(ax, matrix, labels):
    """Minimal heatmap without seaborn."""
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]),
                    ha="center", va="center", fontsize=8,
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    return im


def _plot_model_comparison(results: dict, output_dir: str) -> None:
    names  = list(results.keys())
    accs   = [results[n]["accuracy"]    for n in names]
    f1_mac = [results[n]["f1_macro"]    for n in names]
    f1_wtd = [results[n]["f1_weighted"] for n in names]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width,     accs,   width, label="Accuracy",    color="#4e79a7")
    ax.bar(x,             f1_mac, width, label="F1 Macro",    color="#f28e2b")
    ax.bar(x + width,     f1_wtd, width, label="F1 Weighted", color="#59a14f")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Comparison")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved model_comparison.png")


def _plot_feature_importance(clf, feature_names: list,
                              model_name: str, output_dir: str) -> None:
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(len(importances)),
           importances[indices],
           color="#4e79a7")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices],
                       rotation=30, ha="right")
    ax.set_title(f"Feature Importances – {model_name}")
    ax.set_ylabel("Importance")
    fig.tight_layout()
    path = os.path.join(output_dir, "feature_importances.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved feature_importances.png")


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_preprocessing import load_data, clean_data
    from feature_engineering import prepare_train_test

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    df_raw   = load_data()
    df_clean = clean_data(df_raw)
    X_train, X_test, y_train, y_test, feat_names, _ = prepare_train_test(df_clean)

    results = train_and_evaluate(X_train, X_test, y_train, y_test, feat_names)
    print("\nTraining complete. Check outputs/ for plots and metrics.")
