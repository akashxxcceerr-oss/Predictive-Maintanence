"""
main.py
───────
Entry-point for the Predictive Maintenance pipeline.

Usage
─────
  # Full pipeline (EDA → train → deploy)
  python main.py --pipeline full

  # Only train models locally
  python main.py --pipeline train

  # EDA only
  python main.py --pipeline eda

  # Predict on a CSV of new readings
  python main.py --pipeline predict --input data/new_readings.csv

  # Deploy best model to IBM WML
  python main.py --pipeline deploy

  # Upload dataset to IBM COS
  python main.py --pipeline upload_cos
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# ── Allow running from project root ─────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import load_data, clean_data, generate_eda_plots, check_data_quality
from feature_engineering import prepare_train_test, engineer_features, FEATURE_COLS_ENGINEERED
from model_training import train_and_evaluate
from ibm_watson_ml import IBMWatsonMLClient, IBMCOSClient, predict_local
from evaluate import run_evaluation

load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
os.makedirs(os.getenv("OUTPUT_DIR", "outputs/"), exist_ok=True)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.getenv("LOG_FILE", "outputs/app.log"), mode="a"),
    ],
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
def run_eda() -> pd.DataFrame:
    """Load, validate, clean, and generate EDA plots."""
    logger.info("=== STEP 1 - Data Loading & EDA ===========================")
    df_raw = load_data()
    check_data_quality(df_raw)
    df_clean = clean_data(df_raw)
    generate_eda_plots(df_clean, output_dir=os.getenv("OUTPUT_DIR", "outputs/"))
    logger.info("EDA complete. Plots written to outputs/")
    return df_clean


def run_training(df_clean: pd.DataFrame = None) -> dict:
    """Feature engineering → train / evaluate all classifiers."""
    if df_clean is None:
        df_clean = run_eda()

    logger.info("=== STEP 2 - Feature Engineering & Model Training ==========")
    X_train, X_test, y_train, y_test, feat_names, scaler = prepare_train_test(
        df_clean,
        test_size    = float(os.getenv("TEST_SIZE",    "0.2")),
        random_state = int(os.getenv("RANDOM_STATE", "42")),
    )

    results = train_and_evaluate(X_train, X_test, y_train, y_test, feat_names)

    best = max(results, key=lambda k: results[k]["f1_weighted"])
    logger.info(
        "Best model : %s  |  Accuracy=%.4f  F1-weighted=%.4f",
        best,
        results[best]["accuracy"],
        results[best]["f1_weighted"],
    )
    return results


def run_evaluate() -> None:
    """Advanced evaluation: SMOTE, ROC-AUC, hyperparameter tuning."""
    logger.info("=== STEP 3 - Advanced Evaluation ===========================")
    report = run_evaluation()
    logger.info(
        "Evaluation complete  |  Accuracy=%.4f  F1-weighted=%.4f  ROC-AUC=%.4f",
        report["accuracy"], report["f1_weighted"], report["macro_roc_auc"],
    )


def run_deploy() -> None:
    """Store and deploy the best model to IBM Watson ML."""
    logger.info("=== STEP 3 - IBM Watson ML Deployment ======================")
    wml = IBMWatsonMLClient()
    if wml.client is None:
        logger.error("WML client not available. Check .env credentials.")
        return

    model_uid = wml.store_model(
        model_path  = os.path.join(os.getenv("MODEL_DIR", "models/"), "best_model.joblib"),
        model_name  = "PredictiveMaintenance_Classifier",
        description = "Random Forest multi-class failure type predictor",
    )
    if model_uid:
        wml.deploy_model(model_uid, deployment_name="PM_Online_Deployment")


def run_upload_cos() -> None:
    """Upload dataset and model artefacts to IBM Cloud Object Storage."""
    logger.info("=== IBM COS Upload ==========================================")
    cos = IBMCOSClient()
    if cos.cos is None:
        logger.error("COS client not available. Check .env credentials.")
        return

    cos.ensure_bucket_exists()
    files_to_upload = [
        (os.getenv("DATA_PATH", "data/predictive_maintenance.csv"), "dataset/predictive_maintenance.csv"),
        ("models/best_model.joblib",  "models/best_model.joblib"),
        ("models/scaler.joblib",      "models/scaler.joblib"),
        ("outputs/metrics_summary.json", "outputs/metrics_summary.json"),
    ]
    for local, remote in files_to_upload:
        if os.path.exists(local):
            cos.upload_file(local, remote)
        else:
            logger.warning("File not found (skip): %s", local)


def run_predict(input_csv: str) -> None:
    """Run inference on a CSV of new sensor readings."""
    logger.info("=== Inference on %s =========================================", input_csv)

    df = pd.read_csv(input_csv)

    # Accept raw column names or already-renamed columns
    rename = {
        "Air temperature [K]":     "air_temp_k",
        "Process temperature [K]": "proc_temp_k",
        "Rotational speed [rpm]":  "rot_speed_rpm",
        "Torque [Nm]":             "torque_nm",
        "Tool wear [min]":         "tool_wear_min",
        "Type":                    "machine_type",
    }
    df.rename(columns=rename, inplace=True)

    if "machine_type_encoded" not in df.columns:
        df["machine_type_encoded"] = df["machine_type"].map({"L": 0, "M": 1, "H": 2}).fillna(1)

    df = engineer_features(df)
    X = df[FEATURE_COLS_ENGINEERED].values

    preds = predict_local(
        X,
        model_path  = os.path.join(os.getenv("MODEL_DIR", "models/"), "best_model.joblib"),
        scaler_path = os.path.join(os.getenv("MODEL_DIR", "models/"), "scaler.joblib"),
    )

    out_df = df[["machine_type", "air_temp_k", "proc_temp_k",
                 "rot_speed_rpm", "torque_nm", "tool_wear_min"]].copy()
    out_df["predicted_failure_type"] = [p["label"] for p in preds]
    if "probabilities" in preds[0]:
        for cls_label in preds[0]["probabilities"]:
            out_df[f"prob_{cls_label}"] = [p["probabilities"].get(cls_label, 0) for p in preds]

    out_path = os.path.join(os.getenv("OUTPUT_DIR", "outputs/"), "predictions.csv")
    out_df.to_csv(out_path, index=False)
    logger.info("Predictions saved to %s", out_path)
    print(out_df.head(10).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Predictive Maintenance – IBM Cloud Lite Project"
    )
    parser.add_argument(
        "--pipeline",
        choices=["full", "eda", "train", "evaluate", "deploy", "predict", "upload_cos"],
        default="full",
        help="Pipeline stage to run (default: full)",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to input CSV for --pipeline predict",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Predictive Maintenance Pipeline  |  stage=%s", args.pipeline)

    if args.pipeline == "eda":
        run_eda()

    elif args.pipeline == "train":
        run_training()

    elif args.pipeline == "deploy":
        run_deploy()

    elif args.pipeline == "evaluate":
        run_evaluate()

    elif args.pipeline == "upload_cos":
        run_upload_cos()

    elif args.pipeline == "predict":
        if not args.input:
            logger.error("--input CSV path required for predict pipeline.")
            sys.exit(1)
        run_predict(args.input)

    elif args.pipeline == "full":
        df_clean = run_eda()
        run_training(df_clean)
        run_evaluate()
        logger.info(
            "\nNext steps:\n"
            "  Deploy to IBM Watson ML : python main.py --pipeline deploy\n"
            "  Upload to IBM COS       : python main.py --pipeline upload_cos\n"
            "  Start web UI            : python app.py"
        )

    logger.info("Pipeline stage '%s' finished.", args.pipeline)


if __name__ == "__main__":
    main()
