"""
data_preprocessing.py
─────────────────────
Loads, cleans, and performs exploratory data analysis (EDA) on the
Kaggle Predictive-Maintenance dataset.

Failure types in dataset
  0 – No Failure
  1 – Heat Dissipation Failure
  2 – Overstrain Failure
  3 – Power Failure
  4 – Tool Wear Failure
  5 – Random Failures (rare, excluded by Kaggle labelling)
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server use
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Column aliases (original names contain spaces / brackets) ─────────────────
COLUMN_RENAME = {
    "UDI":                       "udi",
    "Product ID":                "product_id",
    "Type":                      "machine_type",
    "Air temperature [K]":       "air_temp_k",
    "Process temperature [K]":   "proc_temp_k",
    "Rotational speed [rpm]":    "rot_speed_rpm",
    "Torque [Nm]":               "torque_nm",
    "Tool wear [min]":           "tool_wear_min",
    "Target":                    "target",
    "Failure Type":              "failure_type",
}

FAILURE_LABEL_MAP = {
    "No Failure":              0,
    "Heat Dissipation Failure": 1,
    "Overstrain Failure":      2,
    "Power Failure":           3,
    "Tool Wear Failure":       4,
    "Random Failures":         5,
}

FEATURE_COLS = [
    "machine_type_encoded",
    "air_temp_k",
    "proc_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

TARGET_COL = "failure_type_encoded"


# ─────────────────────────────────────────────────────────────────────────────
def load_data(data_path: str | None = None) -> pd.DataFrame:
    """Load CSV from *data_path* (defaults to $DATA_PATH env var)."""
    path = data_path or os.getenv("DATA_PATH", "data/predictive_maintenance.csv")
    logger.info("Loading dataset from: %s", path)
    df = pd.read_csv(path, encoding="utf-8-sig")   # strip BOM if present
    df.rename(columns=COLUMN_RENAME, inplace=True)
    logger.info("Dataset shape: %s", df.shape)
    return df


def check_data_quality(df: pd.DataFrame) -> None:
    """Log data quality metrics."""
    logger.info("-- Data Quality Report ------------------------------")
    logger.info("Shape      : %s", df.shape)
    logger.info("Dtypes:\n%s", df.dtypes.to_string())
    missing = df.isnull().sum()
    if missing.any():
        logger.warning("Missing values detected:\n%s", missing[missing > 0].to_string())
    else:
        logger.info("No missing values found.")
    dupes = df.duplicated().sum()
    logger.info("Duplicate rows: %d", dupes)
    logger.info("Failure type distribution:\n%s",
                df["failure_type"].value_counts().to_string())


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning:
      • Drop duplicates
      • Drop irrelevant identifier columns
      • Encode categorical variables
    """
    df = df.drop_duplicates()

    # Encode machine type  L=0, M=1, H=2
    type_map = {"L": 0, "M": 1, "H": 2}
    df["machine_type_encoded"] = df["machine_type"].map(type_map)

    # Encode failure type to integer label
    df["failure_type_encoded"] = df["failure_type"].map(FAILURE_LABEL_MAP)

    # Drop rows where failure_type is unknown (unmapped → NaN)
    before = len(df)
    df = df.dropna(subset=["failure_type_encoded"])
    df["failure_type_encoded"] = df["failure_type_encoded"].astype(int)
    after = len(df)
    if before != after:
        logger.warning("Dropped %d rows with unrecognised failure type.", before - after)

    logger.info("Clean dataset shape: %s", df.shape)
    return df.reset_index(drop=True)


def generate_eda_plots(df: pd.DataFrame, output_dir: str = "outputs/") -> None:
    """Save EDA visualisations to *output_dir*."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Failure type distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    order = df["failure_type"].value_counts().index
    sns.countplot(data=df, y="failure_type", hue="failure_type",
                  order=order, palette="Set2", legend=False, ax=ax)
    ax.set_title("Failure Type Distribution")
    ax.set_xlabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "failure_distribution.png"), dpi=120)
    plt.close(fig)
    logger.info("Saved failure_distribution.png")

    # 2. Correlation heatmap (numeric features only)
    num_cols = ["air_temp_k", "proc_temp_k", "rot_speed_rpm",
                "torque_nm", "tool_wear_min", "target"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=120)
    plt.close(fig)
    logger.info("Saved correlation_heatmap.png")

    # 3. Rotational speed vs Torque (coloured by failure)
    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        df["rot_speed_rpm"], df["torque_nm"],
        c=df["failure_type_encoded"], cmap="tab10",
        alpha=0.4, s=10,
    )
    plt.colorbar(scatter, ax=ax, label="Failure Type")
    ax.set_xlabel("Rotational Speed (rpm)")
    ax.set_ylabel("Torque (Nm)")
    ax.set_title("Rotational Speed vs Torque by Failure Type")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "speed_vs_torque.png"), dpi=120)
    plt.close(fig)
    logger.info("Saved speed_vs_torque.png")

    # 4. Tool wear distribution per failure type
    fig, ax = plt.subplots(figsize=(10, 5))
    for ft, grp in df.groupby("failure_type"):
        grp["tool_wear_min"].plot.hist(bins=40, alpha=0.6, label=ft, ax=ax)
    ax.set_xlabel("Tool Wear (min)")
    ax.set_title("Tool Wear Distribution by Failure Type")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "tool_wear_by_failure.png"), dpi=120)
    plt.close(fig)
    logger.info("Saved tool_wear_by_failure.png")


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.getenv("LOG_FILE", "outputs/app.log"), mode="a"),
        ],
    )
    df_raw = load_data()
    check_data_quality(df_raw)
    df_clean = clean_data(df_raw)
    generate_eda_plots(df_clean, output_dir=os.getenv("OUTPUT_DIR", "outputs/"))
    print("\nEDA complete. Plots saved to outputs/")
