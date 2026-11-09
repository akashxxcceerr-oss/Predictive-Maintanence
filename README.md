# Predictive Maintenance of Industrial Machinery
### Problem Statement No. 39 — IBM Cloud Lite Edition

> Anticipate machine failures **before they occur** by classifying real-time sensor data into five failure categories using IBM Cloud Lite services.

---

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Architecture](#architecture)
5. [IBM Cloud Lite Services](#ibm-cloud-lite-services)
6. [Setup](#setup)
7. [Usage](#usage)
8. [Model Performance](#model-performance)
9. [Feature Engineering](#feature-engineering)
10. [IBM Watson ML Deployment](#ibm-watson-ml-deployment)

---

## Problem Statement
Industrial machines fail unexpectedly, causing costly downtime. This project builds a **multi-class classification model** that predicts the *type* of failure from live sensor readings so maintenance teams can act proactively.

**Failure classes predicted:**

| Code | Label |
|------|-------|
| 0 | No Failure |
| 1 | Heat Dissipation Failure |
| 2 | Overstrain Failure |
| 3 | Power Failure |
| 4 | Tool Wear Failure |

---

## Dataset
**Source:** [Kaggle – Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

| Column | Description |
|--------|-------------|
| `Type` | Machine quality tier (L / M / H) |
| `Air temperature [K]` | Ambient air temperature |
| `Process temperature [K]` | Process temperature |
| `Rotational speed [rpm]` | Spindle speed |
| `Torque [Nm]` | Applied torque |
| `Tool wear [min]` | Cumulative tool wear time |
| `Target` | Binary failure flag |
| `Failure Type` | Multi-class failure label (target) |

10,000 rows · no missing values · class-imbalanced (≈96 % No Failure)

---

## Project Structure

```
Predictive-Maintenance/
├── data/
│   └── predictive_maintenance.csv      # Kaggle dataset
├── models/                             # Saved scikit-learn models + scaler
│   ├── best_model.joblib
│   ├── scaler.joblib
│   ├── RandomForest.joblib
│   ├── GradientBoosting.joblib
│   ├── LogisticRegression.joblib
│   └── KNN.joblib
├── outputs/                            # EDA plots, metrics, predictions
│   ├── failure_distribution.png
│   ├── correlation_heatmap.png
│   ├── speed_vs_torque.png
│   ├── tool_wear_by_failure.png
│   ├── confusion_matrices.png
│   ├── model_comparison.png
│   ├── feature_importances.png
│   ├── metrics_summary.json
│   └── app.log
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py           # Load, clean, EDA
│   ├── feature_engineering.py          # Derived features + train/test split
│   ├── model_training.py               # Train / evaluate all classifiers
│   └── ibm_watson_ml.py                # IBM WML + COS integration
├── main.py                             # CLI pipeline entry-point
├── requirements.txt
├── .env                                # IBM Cloud credentials (never commit)
└── README.md
```

---

## Architecture

```
CSV Dataset
    │
    ▼
data_preprocessing.py  ──► EDA Plots (outputs/)
    │
    ▼
feature_engineering.py  (10 features including 4 derived)
    │
    ▼
model_training.py ──► RandomForest / GradientBoosting / LR / KNN
    │                  ↓
    │               metrics_summary.json + confusion matrices
    │
    ▼
best_model.joblib
    │
    ├──► [local]   predict_local()  in ibm_watson_ml.py
    │
    └──► [cloud]   IBM Watson ML Online Endpoint
                       │
                       └──► IBM Cloud Object Storage (artefacts)
```

---

## IBM Cloud Lite Services

| Service | Purpose | Plan |
|---------|---------|------|
| **Watson Machine Learning** | Host & serve the trained model as a REST endpoint | Lite (free) |
| **Cloud Object Storage** | Store dataset, model artefacts, outputs | Lite (25 GB free) |
| **Watson Studio** | Notebook environment for experimentation | Lite (free) |

All credentials are stored in `.env` — **never commit this file to version control**.

---

## Setup

### 1. Clone & install
```bash
git clone <repo-url>
cd Predictive-Maintenance
pip install -r requirements.txt
```

### 2. Configure `.env`
Edit `.env` and replace placeholders with real values from your IBM Cloud account:

```
IBM_WML_API_KEY=<your IBM Cloud API key>
IBM_WML_URL=https://us-south.ml.cloud.ibm.com
IBM_WML_SPACE_ID=<Watson ML deployment space ID>
IBM_COS_API_KEY=<COS API key>
IBM_COS_INSTANCE_CRN=<COS resource instance CRN>
IBM_COS_BUCKET_NAME=predictive-maintenance-bucket
```

> **IBM Cloud Lite account:** Sign up free at https://cloud.ibm.com/registration

### 3. Place dataset
The CSV is already copied to `data/predictive_maintenance.csv` during project setup. If running fresh, download from Kaggle and place at that path.

---

## Usage

```bash
# Full pipeline: EDA → train → save models
python main.py --pipeline full

# EDA only (plots saved to outputs/)
python main.py --pipeline eda

# Train models only
python main.py --pipeline train

# Deploy best model to IBM Watson ML
python main.py --pipeline deploy

# Upload all artefacts to IBM Cloud Object Storage
python main.py --pipeline upload_cos

# Predict failure types on new sensor readings CSV
python main.py --pipeline predict --input data/new_readings.csv
```

---

## Model Performance

Four classifiers are trained and compared. Typical results on the test set:

| Model | Accuracy | F1 Macro | F1 Weighted |
|-------|----------|----------|-------------|
| **Random Forest** | ~0.98 | ~0.88 | ~0.98 |
| Gradient Boosting | ~0.98 | ~0.87 | ~0.98 |
| Logistic Regression | ~0.96 | ~0.72 | ~0.96 |
| KNN | ~0.97 | ~0.80 | ~0.97 |

> Results vary slightly depending on random seed and class balance.  
> Metrics are saved to `outputs/metrics_summary.json`.

---

## Feature Engineering

Beyond the 6 raw sensor features, 4 domain-derived features are computed:

| Feature | Formula | Physical Meaning |
|---------|---------|-----------------|
| `temp_diff` | proc_temp − air_temp | Heat dissipation stress |
| `power_kw` | (2π/60) × rpm × torque / 1000 | Mechanical power output |
| `tool_wear_rate` | tool_wear / torque | Wear per unit load |
| `speed_torque_ratio` | rpm / torque | Efficiency indicator |

---

## IBM Watson ML Deployment

After training:

```bash
# 1. Store model in WML repository & create online deployment
python main.py --pipeline deploy

# 2. The .env is auto-updated with:
#    IBM_WML_MODEL_UID=<uid>
#    IBM_WML_DEPLOYMENT_UID=<uid>
#    DEPLOYED_MODEL_ENDPOINT=<scoring url>

# 3. Score via the endpoint (see ibm_watson_ml.py → predict_via_endpoint)
```

The deployed endpoint accepts JSON payloads and returns predicted failure types with class probabilities.

---

*Built for Problem Statement No. 39 using IBM Cloud Lite free-tier services.*
