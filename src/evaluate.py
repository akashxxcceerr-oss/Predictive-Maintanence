"""
evaluate.py
───────────
Advanced model evaluation module.

Produces:
  • ROC-AUC curves (one-vs-rest per class)
  • Per-class precision / recall / F1 bar charts
  • SMOTE-balanced retraining comparison
  • Hyperparameter tuning via GridSearchCV
  • Final evaluation report saved to outputs/evaluation_report.json

Run standalone:
  python src/evaluate.py
"""

import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from data_preprocessing import load_data, clean_data
from feature_engineering import prepare_train_test, FEATURE_COLS_ENGINEERED

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs/")
MODEL_DIR  = os.getenv("MODEL_DIR",  "models/")
RS         = int(os.getenv("RANDOM_STATE", "42"))

LABEL_NAMES = [
    "No Failure",
    "Heat Dissipation",
    "Overstrain",
    "Power Failure",
    "Tool Wear",
    "Random Failure",
]


# ─────────────────────────────────────────────────────────────────────────────
def smote_oversample(X_train: np.ndarray, y_train: np.ndarray):
    """
    Apply SMOTE to handle class imbalance.
    Falls back to original data if imbalanced-learn is not installed.
    """
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=RS, k_neighbors=3)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        before = pd.Series(y_train).value_counts().sort_index().to_dict()
        after  = pd.Series(y_res).value_counts().sort_index().to_dict()
        logger.info("SMOTE applied. Before: %s  After: %s", before, after)
        return X_res, y_res, True
    except ImportError:
        logger.warning(
            "imbalanced-learn not installed. Skipping SMOTE. "
            "Install with: pip install imbalanced-learn"
        )
        return X_train, y_train, False


# ─────────────────────────────────────────────────────────────────────────────
def hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "GradientBoosting",
) -> object:
    """
    Run GridSearchCV on *model_name* and return the best estimator.
    """
    param_grids = {
        "GradientBoosting": {
            "clf": GradientBoostingClassifier(random_state=RS),
            "grid": {
                "n_estimators":  [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth":     [3, 5],
            },
        },
        "RandomForest": {
            "clf": RandomForestClassifier(
                class_weight="balanced", random_state=RS, n_jobs=-1
            ),
            "grid": {
                "n_estimators":       [100, 200],
                "max_depth":          [None, 10],
                "min_samples_split":  [2, 5],
            },
        },
    }

    if model_name not in param_grids:
        logger.warning("No grid defined for %s. Using GradientBoosting.", model_name)
        model_name = "GradientBoosting"

    entry = param_grids[model_name]
    logger.info("Running GridSearchCV for %s ...", model_name)

    gs = GridSearchCV(
        entry["clf"],
        entry["grid"],
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train, y_train)

    logger.info("Best params : %s", gs.best_params_)
    logger.info("Best CV F1  : %.4f", gs.best_score_)
    return gs.best_estimator_


# ─────────────────────────────────────────────────────────────────────────────
def plot_roc_curves(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
) -> float:
    """
    Plot one-vs-rest ROC curves for each class.
    Returns the macro-average ROC-AUC.
    """
    classes    = sorted(np.unique(y_test))
    y_test_bin = label_binarize(y_test, classes=list(range(len(LABEL_NAMES))))
    y_score    = clf.predict_proba(X_test)

    colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2", "#b07aa1"]

    fig, ax = plt.subplots(figsize=(9, 6))
    macro_auc_sum = 0.0
    valid_classes = 0

    for i in classes:
        if y_test_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        macro_auc_sum += roc_auc
        valid_classes += 1
        ax.plot(
            fpr, tpr,
            color=colors[i % len(colors)], lw=2,
            label=f"{LABEL_NAMES[i]}  (AUC={roc_auc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest) – Tuned Best Model")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = os.path.join(output_dir, "roc_curves.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved roc_curves.png")

    macro_auc = macro_auc_sum / valid_classes if valid_classes > 0 else 0.0
    logger.info("Macro ROC-AUC: %.4f", macro_auc)
    return macro_auc


# ─────────────────────────────────────────────────────────────────────────────
def plot_per_class_metrics(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str,
) -> None:
    """Bar chart of per-class precision, recall, F1."""
    y_pred = clf.predict(X_test)
    present = sorted(np.unique(np.concatenate([y_test, y_pred])))
    labels  = [LABEL_NAMES[i] for i in present]

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=present, zero_division=0
    )

    x     = np.arange(len(labels))
    width = 0.28

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, prec, width, label="Precision", color="#4e79a7")
    ax.bar(x,         rec,  width, label="Recall",    color="#f28e2b")
    ax.bar(x + width, f1,   width, label="F1 Score",  color="#59a14f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-Class Precision / Recall / F1 – Tuned Best Model")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "per_class_metrics.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved per_class_metrics.png")


# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation() -> dict:
    """Full evaluation pipeline. Returns the evaluation report dict."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR,  exist_ok=True)

    # 1. Load & prepare data
    logger.info("=== Loading data ===")
    df_raw   = load_data()
    df_clean = clean_data(df_raw)
    X_train, X_test, y_train, y_test, feat_names, scaler = prepare_train_test(df_clean)

    # 2. SMOTE oversampling
    logger.info("=== SMOTE oversampling ===")
    X_train_bal, y_train_bal, smote_used = smote_oversample(X_train, y_train)

    # 3. Pick which model to tune (load best from previous training)
    best_model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    metrics_path    = os.path.join(OUTPUT_DIR, "metrics_summary.json")

    best_model_name = "GradientBoosting"
    if os.path.exists(metrics_path):
        with open(metrics_path) as fh:
            meta = json.load(fh)
        best_model_name = meta.get("best_model", "GradientBoosting")

    # 4. Hyperparameter tuning
    logger.info("=== Hyperparameter tuning: %s ===", best_model_name)
    tuned_clf = hyperparameter_tuning(X_train_bal, y_train_bal, best_model_name)

    # 5. Evaluate tuned model
    y_pred = tuned_clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1_w   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_m   = f1_score(y_test, y_pred, average="macro",    zero_division=0)

    present = sorted(np.unique(np.concatenate([y_test, y_pred])))
    report  = classification_report(
        y_test, y_pred,
        labels=present,
        target_names=[LABEL_NAMES[i] for i in present],
        zero_division=0,
    )
    logger.info("Tuned model evaluation:\n%s", report)

    # 6. ROC curves
    macro_auc = plot_roc_curves(tuned_clf, X_test, y_test, OUTPUT_DIR)

    # 7. Per-class metrics chart
    plot_per_class_metrics(tuned_clf, X_test, y_test, OUTPUT_DIR)

    # 8. Save tuned model
    tuned_path = os.path.join(MODEL_DIR, "best_model_tuned.joblib")
    joblib.dump(tuned_clf, tuned_path)
    logger.info("Tuned model saved -> %s", tuned_path)

    # 9. Compile evaluation report
    eval_report = {
        "model_name":   best_model_name,
        "smote_used":   smote_used,
        "accuracy":     round(acc,       4),
        "f1_macro":     round(f1_m,      4),
        "f1_weighted":  round(f1_w,      4),
        "macro_roc_auc": round(macro_auc, 4),
        "classification_report": report,
    }

    report_path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
    with open(report_path, "w") as fh:
        json.dump(
            {k: v for k, v in eval_report.items() if k != "classification_report"},
            fh, indent=2,
        )
    logger.info("Evaluation report saved -> %s", report_path)

    return eval_report


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    )
    report = run_evaluation()
    print("\n=== Evaluation Complete ===")
    print(f"  Accuracy     : {report['accuracy']:.4f}")
    print(f"  F1-Weighted  : {report['f1_weighted']:.4f}")
    print(f"  F1-Macro     : {report['f1_macro']:.4f}")
    print(f"  Macro ROC-AUC: {report['macro_roc_auc']:.4f}")
    print(f"  SMOTE used   : {report['smote_used']}")
    print(f"\nOutputs saved to {OUTPUT_DIR}")
