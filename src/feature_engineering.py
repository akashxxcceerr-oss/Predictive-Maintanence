"""
feature_engineering.py
──────────────────────
Creates domain-informed features from raw sensor readings and
prepares train / test splits ready for model training.

New derived features:
  • temp_diff          – Process temp minus air temp (heat dissipation proxy)
  • power_kw           – Rotational speed × Torque (approx. mechanical power)
  • tool_wear_rate     – Tool wear per unit torque
  • speed_torque_ratio – Speed / Torque ratio
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

FEATURE_COLS_ENGINEERED = [
    "machine_type_encoded",
    "air_temp_k",
    "proc_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
    "temp_diff",
    "power_kw",
    "tool_wear_rate",
    "speed_torque_ratio",
]

TARGET_COL = "failure_type_encoded"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived sensor features to *df* in-place and return it."""
    df = df.copy()

    # Temperature differential (key indicator of heat dissipation failure)
    df["temp_diff"] = df["proc_temp_k"] - df["air_temp_k"]

    # Mechanical power (W → kW)
    # P ≈ (2π / 60) × N × T  [simplified to N×T here for relative magnitude]
    df["power_kw"] = (df["rot_speed_rpm"] * df["torque_nm"] * 2 * np.pi / 60) / 1000

    # Tool wear per unit torque
    df["tool_wear_rate"] = df["tool_wear_min"] / (df["torque_nm"] + 1e-6)

    # Speed-to-torque ratio (low torque at high speed → power failure risk)
    df["speed_torque_ratio"] = df["rot_speed_rpm"] / (df["torque_nm"] + 1e-6)

    logger.info("Feature engineering complete. Final columns: %s",
                list(df.columns))
    return df


def prepare_train_test(
    df: pd.DataFrame,
    test_size: float = None,
    random_state: int = None,
    scaler_path: str = "models/scaler.joblib",
) -> tuple:
    """
    Split into train/test, fit a StandardScaler on train features.

    Returns
    -------
    X_train, X_test, y_train, y_test : np.ndarray
    feature_names                     : list[str]
    scaler                            : fitted StandardScaler
    """
    if test_size is None:
        test_size    = float(os.getenv("TEST_SIZE",    "0.2"))
    if random_state is None:
        random_state = int(os.getenv("RANDOM_STATE", "42"))

    df_eng = engineer_features(df)

    missing_cols = [c for c in FEATURE_COLS_ENGINEERED if c not in df_eng.columns]
    if missing_cols:
        raise ValueError(f"Missing columns after feature engineering: {missing_cols}")

    X = df_eng[FEATURE_COLS_ENGINEERED].values
    y = df_eng[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Persist scaler for inference
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved to %s", scaler_path)

    logger.info(
        "Train size: %d  |  Test size: %d  |  Classes: %s",
        len(X_train), len(X_test), np.unique(y),
    )
    return X_train, X_test, y_train, y_test, FEATURE_COLS_ENGINEERED, scaler


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_preprocessing import load_data, clean_data

    df_raw   = load_data()
    df_clean = clean_data(df_raw)
    X_train, X_test, y_train, y_test, feat_names, _ = prepare_train_test(df_clean)
    print(f"X_train shape : {X_train.shape}")
    print(f"X_test  shape : {X_test.shape}")
    print(f"Features      : {feat_names}")
