"""
app.py
──────
Flask web application for real-time Predictive Maintenance inference.

Provides:
  GET  /            – Dashboard UI showing model metrics
  POST /predict     – JSON API: accept sensor readings, return failure prediction
  GET  /health      – Health check endpoint

Run locally:
  python app.py

Access at: http://localhost:5000
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template_string
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from feature_engineering import engineer_features, FEATURE_COLS_ENGINEERED

load_dotenv()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Load model & scaler once at startup ───────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "models/")

def _load_artifacts():
    model_path  = os.path.join(MODEL_DIR, "best_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run: python main.py --pipeline train"
        )
    clf    = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info("Model and scaler loaded from %s", MODEL_DIR)
    return clf, scaler

try:
    CLF, SCALER = _load_artifacts()
    MODEL_READY = True
except Exception as exc:
    logger.error("Failed to load model artifacts: %s", exc)
    CLF, SCALER = None, None
    MODEL_READY = False

LABEL_MAP = {
    0: "No Failure",
    1: "Heat Dissipation Failure",
    2: "Overstrain Failure",
    3: "Power Failure",
    4: "Tool Wear Failure",
    5: "Random Failures",
}

# ── Load metrics summary if available ─────────────────────────────────────────
def _load_metrics():
    path = os.path.join(os.getenv("OUTPUT_DIR", "outputs/"), "metrics_summary.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

METRICS = _load_metrics()

# ─────────────────────────────────────────────────────────────────────────────
# HTML template (single-file, no external assets)
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Predictive Maintenance Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, "Segoe UI", sans-serif; background: #f0f4f8; color: #1f2328; }
  header { background: #0f3460; color: #fff; padding: 18px 32px; display: flex; align-items: center; gap: 16px; }
  header h1 { font-size: 1.35rem; font-weight: 600; }
  header span { font-size: 0.85rem; opacity: 0.75; }
  .container { max-width: 1100px; margin: 32px auto; padding: 0 20px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .card { background: #fff; border-radius: 10px; padding: 24px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
  .card h2 { font-size: 1.05rem; font-weight: 600; margin-bottom: 16px; color: #0f3460; border-bottom: 2px solid #e5e7eb; padding-bottom: 8px; }
  label { display: block; font-size: 0.82rem; font-weight: 600; color: #57606a; margin-bottom: 4px; margin-top: 12px; }
  input, select { width: 100%; padding: 8px 10px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 0.9rem; }
  input:focus, select:focus { outline: 2px solid #3b82d4; border-color: transparent; }
  .btn { margin-top: 20px; width: 100%; padding: 11px; background: #0f3460; color: #fff;
         border: none; border-radius: 7px; font-size: 1rem; font-weight: 600; cursor: pointer; }
  .btn:hover { background: #1a4a8a; }
  .btn:disabled { background: #9ca3af; cursor: not-allowed; }
  #result { margin-top: 20px; padding: 16px; border-radius: 8px; display: none; }
  .result-ok  { background: #d1fae5; border-left: 4px solid #10b981; }
  .result-warn{ background: #fef3c7; border-left: 4px solid #f59e0b; }
  .result-fail{ background: #fee2e2; border-left: 4px solid #ef4444; }
  .result-label { font-size: 1.2rem; font-weight: 700; }
  .prob-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; font-size: 0.82rem; }
  .prob-bar-wrap { flex: 1; background: #e5e7eb; border-radius: 4px; height: 10px; }
  .prob-bar { height: 10px; border-radius: 4px; background: #3b82d4; }
  table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
  th { background: #f7f8fa; padding: 8px 12px; text-align: left; border-bottom: 2px solid #e5e7eb; }
  td { padding: 8px 12px; border-bottom: 1px solid #f0f0f0; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; }
  .badge-green { background: #d1fae5; color: #065f46; }
  .badge-blue  { background: #dbeafe; color: #1e40af; }
  .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 6px; }
  .dot-green { background: #10b981; }
  .dot-red   { background: #ef4444; }
  @media(max-width:720px){ .grid2{grid-template-columns:1fr;} }
</style>
</head>
<body>
<header>
  <div>
    <h1>Predictive Maintenance Dashboard</h1>
    <span>Problem Statement No. 39 &nbsp;|&nbsp; IBM Cloud Lite Edition</span>
  </div>
</header>

<div class="container">

  <!-- Status bar -->
  <div class="card" style="margin-bottom:24px; padding:14px 24px; display:flex; gap:32px; align-items:center;">
    <div>
      <span class="status-dot {{ 'dot-green' if model_ready else 'dot-red' }}"></span>
      <strong>Model:</strong> {{ 'Ready' if model_ready else 'Not Loaded' }}
    </div>
    <div><strong>Best Model:</strong> {{ best_model }}</div>
    <div><strong>Accuracy:</strong> {{ accuracy }}</div>
    <div><strong>F1-Weighted:</strong> {{ f1_weighted }}</div>
  </div>

  <div class="grid2">

    <!-- Prediction form -->
    <div class="card">
      <h2>Real-Time Failure Prediction</h2>
      <form id="predict-form">
        <label>Machine Type</label>
        <select name="machine_type" required>
          <option value="L">L – Low quality</option>
          <option value="M" selected>M – Medium quality</option>
          <option value="H">H – High quality</option>
        </select>

        <label>Air Temperature (K)</label>
        <input type="number" step="0.1" name="air_temp_k" value="298.5" required>

        <label>Process Temperature (K)</label>
        <input type="number" step="0.1" name="proc_temp_k" value="308.7" required>

        <label>Rotational Speed (rpm)</label>
        <input type="number" step="1" name="rot_speed_rpm" value="1551" required>

        <label>Torque (Nm)</label>
        <input type="number" step="0.1" name="torque_nm" value="42.8" required>

        <label>Tool Wear (min)</label>
        <input type="number" step="1" name="tool_wear_min" value="0" required>

        <button class="btn" type="submit" id="predict-btn">Predict Failure Type</button>
      </form>

      <div id="result"></div>
    </div>

    <!-- Model metrics -->
    <div class="card">
      <h2>Model Performance</h2>
      <table>
        <thead><tr><th>Model</th><th>Accuracy</th><th>F1 Weighted</th><th>CV Mean</th></tr></thead>
        <tbody>
          {% for name, m in metrics.items() %}
          {% if name != 'best_model' %}
          <tr>
            <td>
              {{ name }}
              {% if name == best_model %}<span class="badge badge-green">Best</span>{% endif %}
            </td>
            <td>{{ "%.4f"|format(m.accuracy) }}</td>
            <td>{{ "%.4f"|format(m.f1_weighted) }}</td>
            <td>{{ "%.4f"|format(m.cv_mean) }}</td>
          </tr>
          {% endif %}
          {% endfor %}
        </tbody>
      </table>

      <h2 style="margin-top:24px;">Failure Type Reference</h2>
      <table>
        <thead><tr><th>Code</th><th>Label</th><th>Type</th></tr></thead>
        <tbody>
          <tr><td>0</td><td>No Failure</td><td><span class="badge badge-green">Normal</span></td></tr>
          <tr><td>1</td><td>Heat Dissipation Failure</td><td><span class="badge badge-blue">Thermal</span></td></tr>
          <tr><td>2</td><td>Overstrain Failure</td><td><span class="badge badge-blue">Mechanical</span></td></tr>
          <tr><td>3</td><td>Power Failure</td><td><span class="badge badge-blue">Electrical</span></td></tr>
          <tr><td>4</td><td>Tool Wear Failure</td><td><span class="badge badge-blue">Wear</span></td></tr>
        </tbody>
      </table>
    </div>

  </div>
</div>

<script>
const LABEL_COLORS = {
  'No Failure': 'result-ok',
  'Heat Dissipation Failure': 'result-warn',
  'Overstrain Failure': 'result-warn',
  'Power Failure': 'result-fail',
  'Tool Wear Failure': 'result-warn',
  'Random Failures': 'result-warn',
};

document.getElementById('predict-form').addEventListener('submit', async function(e) {
  e.preventDefault();
  const btn = document.getElementById('predict-btn');
  btn.disabled = true;
  btn.textContent = 'Predicting...';

  const data = Object.fromEntries(new FormData(this).entries());
  ['air_temp_k','proc_temp_k','rot_speed_rpm','torque_nm','tool_wear_min'].forEach(k => {
    data[k] = parseFloat(data[k]);
  });

  try {
    const resp = await fetch('/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data)
    });
    const result = await resp.json();

    if (result.error) {
      showResult('result-fail', 'Error', result.error, {});
    } else {
      const cls = LABEL_COLORS[result.label] || 'result-warn';
      showResult(cls, result.label, `Class ID: ${result.class_id}`, result.probabilities || {});
    }
  } catch (err) {
    showResult('result-fail', 'Request failed', err.message, {});
  } finally {
    btn.disabled = false;
    btn.textContent = 'Predict Failure Type';
  }
});

function showResult(cls, label, subtitle, probs) {
  const div = document.getElementById('result');
  div.className = cls;
  div.style.display = 'block';

  let probHtml = '';
  const entries = Object.entries(probs).sort((a,b) => b[1]-a[1]);
  entries.forEach(([lbl, p]) => {
    const pct = (p * 100).toFixed(1);
    probHtml += `<div class="prob-row">
      <span style="width:190px;flex-shrink:0">${lbl}</span>
      <div class="prob-bar-wrap"><div class="prob-bar" style="width:${pct}%"></div></div>
      <span style="width:42px;text-align:right">${pct}%</span>
    </div>`;
  });

  div.innerHTML = `
    <div class="result-label">${label}</div>
    <div style="font-size:.82rem;color:#57606a;margin-bottom:10px">${subtitle}</div>
    ${probHtml}
  `;
}
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    best_model = METRICS.get("best_model", "—")
    best_m     = METRICS.get(best_model, {})
    return render_template_string(
        DASHBOARD_HTML,
        model_ready  = MODEL_READY,
        best_model   = best_model,
        accuracy     = f"{best_m.get('accuracy', 0):.4f}",
        f1_weighted  = f"{best_m.get('f1_weighted', 0):.4f}",
        metrics      = METRICS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    JSON body (all fields required):
      {
        "machine_type":   "M",      // L | M | H
        "air_temp_k":     298.5,
        "proc_temp_k":    308.7,
        "rot_speed_rpm":  1551,
        "torque_nm":      42.8,
        "tool_wear_min":  0
      }
    """
    if not MODEL_READY:
        return jsonify({"error": "Model not loaded. Run: python main.py --pipeline train"}), 503

    body = request.get_json(force=True, silent=True)
    if not body:
        return jsonify({"error": "Invalid JSON body"}), 400

    required = ["machine_type", "air_temp_k", "proc_temp_k",
                "rot_speed_rpm", "torque_nm", "tool_wear_min"]
    missing = [f for f in required if f not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        row = pd.DataFrame([{
            "machine_type":  body["machine_type"].upper(),
            "air_temp_k":    float(body["air_temp_k"]),
            "proc_temp_k":   float(body["proc_temp_k"]),
            "rot_speed_rpm": float(body["rot_speed_rpm"]),
            "torque_nm":     float(body["torque_nm"]),
            "tool_wear_min": float(body["tool_wear_min"]),
        }])

        # Encode machine type
        row["machine_type_encoded"] = row["machine_type"].map(
            {"L": 0, "M": 1, "H": 2}
        ).fillna(1)

        # Derive features
        row = engineer_features(row)
        X   = row[FEATURE_COLS_ENGINEERED].values
        X_sc = SCALER.transform(X)

        pred   = int(CLF.predict(X_sc)[0])
        label  = LABEL_MAP.get(pred, "Unknown")

        response = {"class_id": pred, "label": label}

        if hasattr(CLF, "predict_proba"):
            proba = CLF.predict_proba(X_sc)[0]
            response["probabilities"] = {
                LABEL_MAP.get(i, str(i)): round(float(p), 4)
                for i, p in enumerate(proba)
            }

        logger.info("Prediction: %s (%d)  |  input=%s", label, pred, body)
        return jsonify(response)

    except Exception as exc:
        logger.error("Prediction error: %s", exc, exc_info=True)
        return jsonify({"error": str(exc)}), 500


@app.route("/health")
def health():
    return jsonify({
        "status":      "ok" if MODEL_READY else "degraded",
        "model_ready": MODEL_READY,
        "best_model":  METRICS.get("best_model", "unknown"),
    })


@app.route("/metrics")
def metrics_endpoint():
    return jsonify(METRICS)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting Predictive Maintenance API on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=debug)
