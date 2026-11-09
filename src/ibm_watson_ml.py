"""
ibm_watson_ml.py
────────────────
IBM Cloud Lite integration layer.

Responsibilities
  1. Upload the trained model to IBM Watson Machine Learning (WML)
  2. Create a deployment (online endpoint) on the WML deployment space
  3. Score new observations via the deployed endpoint
  4. (Optional) Store the dataset artefact in IBM Cloud Object Storage (COS)

IBM Cloud Lite services used
  • Watson Machine Learning  – free Lite plan
  • Cloud Object Storage     – free Lite tier (25 GB)

All credentials are read from environment variables defined in .env.
"""

import os
import logging
import json
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Optional imports – gracefully skip if ibm_watson_machine_learning not
#    installed (allows local dev without IBM SDK).
try:
    from ibm_watson_machine_learning import APIClient
    WML_AVAILABLE = True
except ImportError:
    WML_AVAILABLE = False
    logger.warning(
        "ibm_watson_machine_learning package not installed. "
        "IBM WML features will be disabled."
    )

# ── Optional COS import ──────────────────────────────────────────────────────
try:
    import ibm_boto3
    from ibm_botocore.client import Config, ClientError
    COS_AVAILABLE = True
except ImportError:
    COS_AVAILABLE = False
    logger.warning(
        "ibm-cos-sdk package not installed. "
        "IBM COS features will be disabled."
    )

LABEL_NAMES = {
    0: "No Failure",
    1: "Heat Dissipation Failure",
    2: "Overstrain Failure",
    3: "Power Failure",
    4: "Tool Wear Failure",
    5: "Random Failures",
}

FEATURE_NAMES = [
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


# ─────────────────────────────────────────────────────────────────────────────
class IBMWatsonMLClient:
    """Thin wrapper around the IBM WML Python client."""

    def __init__(self):
        self.api_key     = os.getenv("IBM_WML_API_KEY")
        self.wml_url     = os.getenv("IBM_WML_URL",
                                     "https://us-south.ml.cloud.ibm.com")
        self.space_id    = os.getenv("IBM_WML_SPACE_ID")
        self.instance_id = os.getenv("IBM_WML_INSTANCE_ID")
        self.client      = None

        if not WML_AVAILABLE:
            logger.error("ibm_watson_machine_learning SDK not available.")
            return
        if not all([self.api_key, self.wml_url, self.space_id]):
            logger.error(
                "Missing IBM WML credentials. Set IBM_WML_API_KEY, "
                "IBM_WML_URL, IBM_WML_SPACE_ID in .env"
            )
            return
        self._connect()

    def _connect(self) -> None:
        """Authenticate and connect to IBM Watson ML."""
        wml_credentials = {
            "apikey": self.api_key,
            "url":    self.wml_url,
        }
        self.client = APIClient(wml_credentials)
        self.client.set.default_space(self.space_id)
        logger.info("Connected to IBM Watson ML at %s", self.wml_url)

    # ── Model store & deploy ─────────────────────────────────────────────────
    def store_model(
        self,
        model_path: str = "models/best_model.joblib",
        model_name: str = "PredictiveMaintenance_RandomForest",
        description: str = "Multi-class failure type classifier",
    ) -> str | None:
        """
        Store a persisted scikit-learn model in the WML repository.

        Returns the model UID on success, None on failure.
        """
        if not self.client:
            logger.error("WML client not initialised.")
            return None

        logger.info("Storing model %s from %s …", model_name, model_path)

        model_props = {
            self.client.repository.ModelMetaNames.NAME:        model_name,
            self.client.repository.ModelMetaNames.DESCRIPTION: description,
            self.client.repository.ModelMetaNames.TYPE:        "scikit-learn_1.1",
            self.client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:
                self.client.software_specifications.get_uid_by_name(
                    "runtime-23.1-py3.10"
                ),
        }

        clf = joblib.load(model_path)
        model_details = self.client.repository.store_model(
            model=clf,
            meta_props=model_props,
            training_data=None,
            training_target=None,
        )
        model_uid = self.client.repository.get_model_id(model_details)
        logger.info("Model stored. UID: %s", model_uid)

        # Persist UID for later use
        self._save_uid("IBM_WML_MODEL_UID", model_uid)
        return model_uid

    def deploy_model(
        self,
        model_uid: str,
        deployment_name: str = "PredictiveMaintenance_Deployment",
    ) -> str | None:
        """
        Create an online deployment for *model_uid*.

        Returns the deployment UID.
        """
        if not self.client:
            return None

        logger.info("Deploying model %s …", model_uid)
        deploy_props = {
            self.client.deployments.ConfigurationMetaNames.NAME:
                deployment_name,
            self.client.deployments.ConfigurationMetaNames.ONLINE: {},
        }
        deployment_details = self.client.deployments.create(
            artifact_uid=model_uid,
            meta_props=deploy_props,
        )
        deployment_uid = self.client.deployments.get_uid(deployment_details)
        scoring_url    = self.client.deployments.get_scoring_href(deployment_details)

        logger.info("Deployment UID : %s", deployment_uid)
        logger.info("Scoring URL    : %s", scoring_url)

        self._save_uid("IBM_WML_DEPLOYMENT_UID", deployment_uid)
        self._save_uid("DEPLOYED_MODEL_ENDPOINT", scoring_url)
        return deployment_uid

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict_via_endpoint(
        self,
        deployment_uid: str,
        X: np.ndarray,
    ) -> list:
        """
        Score *X* (shape n × features) via the WML online endpoint.

        Returns list of predicted class labels (int).
        """
        if not self.client:
            return []

        payload = {
            self.client.deployments.ScoringMetaNames.INPUT_DATA: [
                {"values": X.tolist()}
            ]
        }
        response = self.client.deployments.score(deployment_uid, payload)
        predictions = [p[0] for p in response["predictions"][0]["values"]]
        logger.info("Received %d predictions from WML endpoint.", len(predictions))
        return predictions

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _save_uid(key: str, value: str, env_path: str = ".env") -> None:
        """Append or update a key=value line in the .env file."""
        try:
            with open(env_path, "r") as fh:
                lines = fh.readlines()
            updated = False
            for i, line in enumerate(lines):
                if line.startswith(key + "="):
                    lines[i] = f"{key}={value}\n"
                    updated = True
                    break
            if not updated:
                lines.append(f"{key}={value}\n")
            with open(env_path, "w") as fh:
                fh.writelines(lines)
            logger.info("Updated .env: %s=%s", key, value)
        except OSError as exc:
            logger.warning("Could not update .env: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
class IBMCOSClient:
    """Upload/download dataset artefacts to IBM Cloud Object Storage."""

    def __init__(self):
        self.api_key       = os.getenv("IBM_COS_API_KEY")
        self.crn           = os.getenv("IBM_COS_INSTANCE_CRN")
        self.endpoint      = os.getenv("IBM_COS_ENDPOINT",
                                       "https://s3.us-south.cloud-object-storage"
                                       ".appdomain.cloud")
        self.auth_endpoint = os.getenv("IBM_COS_AUTH_ENDPOINT",
                                       "https://iam.cloud.ibm.com/oidc/token")
        self.bucket        = os.getenv("IBM_COS_BUCKET_NAME",
                                       "predictive-maintenance-bucket")
        self.cos           = None

        if not COS_AVAILABLE:
            logger.error("ibm-cos-sdk not installed.")
            return
        if not all([self.api_key, self.crn]):
            logger.error(
                "Missing COS credentials. Set IBM_COS_API_KEY and "
                "IBM_COS_INSTANCE_CRN in .env"
            )
            return
        self._connect()

    def _connect(self) -> None:
        self.cos = ibm_boto3.client(
            "s3",
            ibm_api_key_id=self.api_key,
            ibm_service_instance_id=self.crn,
            ibm_auth_endpoint=self.auth_endpoint,
            config=Config(signature_version="oauth"),
            endpoint_url=self.endpoint,
        )
        logger.info("Connected to IBM COS bucket: %s", self.bucket)

    def upload_file(self, local_path: str, object_name: str = None) -> bool:
        """Upload *local_path* to the configured COS bucket."""
        if not self.cos:
            return False
        object_name = object_name or os.path.basename(local_path)
        try:
            self.cos.upload_file(local_path, self.bucket, object_name)
            logger.info("Uploaded %s → cos://%s/%s",
                        local_path, self.bucket, object_name)
            return True
        except ClientError as exc:
            logger.error("COS upload failed: %s", exc)
            return False

    def download_file(self, object_name: str, local_path: str) -> bool:
        """Download *object_name* from COS to *local_path*."""
        if not self.cos:
            return False
        try:
            self.cos.download_file(self.bucket, object_name, local_path)
            logger.info("Downloaded cos://%s/%s → %s",
                        self.bucket, object_name, local_path)
            return True
        except ClientError as exc:
            logger.error("COS download failed: %s", exc)
            return False

    def ensure_bucket_exists(self) -> None:
        """Create the bucket if it does not already exist."""
        if not self.cos:
            return
        try:
            self.cos.head_bucket(Bucket=self.bucket)
            logger.info("Bucket '%s' already exists.", self.bucket)
        except ClientError:
            self.cos.create_bucket(Bucket=self.bucket)
            logger.info("Bucket '%s' created.", self.bucket)


# ── Standalone helper: local inference (no WML endpoint needed) ───────────────
def predict_local(
    X: np.ndarray,
    model_path: str = "models/best_model.joblib",
    scaler_path: str = "models/scaler.joblib",
) -> list[dict]:
    """
    Predict failure type using the locally saved model.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_raw_features)
        Raw (unscaled) feature values in the order defined by FEATURE_NAMES.

    Returns
    -------
    List of dicts: {"class_id": int, "label": str, "probabilities": dict}
    """
    clf    = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    X_sc   = scaler.transform(X)

    y_pred  = clf.predict(X_sc)
    y_proba = clf.predict_proba(X_sc) if hasattr(clf, "predict_proba") else None

    results = []
    for i, cls_id in enumerate(y_pred):
        entry = {
            "class_id": int(cls_id),
            "label":    LABEL_NAMES.get(int(cls_id), "Unknown"),
        }
        if y_proba is not None:
            entry["probabilities"] = {
                LABEL_NAMES.get(j, str(j)): round(float(p), 4)
                for j, p in enumerate(y_proba[i])
            }
        results.append(entry)
    return results


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")

    # Demo: local inference on a synthetic sample
    sample = np.array([[
        1,         # machine_type_encoded  (M=1)
        298.5,     # air_temp_k
        308.7,     # proc_temp_k
        1600,      # rot_speed_rpm
        40.0,      # torque_nm
        120,       # tool_wear_min
        10.2,      # temp_diff
        6.7,       # power_kw
        3.0,       # tool_wear_rate
        40.0,      # speed_torque_ratio
    ]])

    predictions = predict_local(sample)
    print("\nLocal inference result:")
    print(json.dumps(predictions, indent=2))
