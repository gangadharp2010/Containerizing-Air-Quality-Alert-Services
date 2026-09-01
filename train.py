"""Fits the alert model and writes model/model.joblib and model/meta.json.

    python train.py

The service only loads what this script wrote. It does not need this file or
data/ at run time.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
FEATURES = ["pm25_today", "wind_speed_ms", "humidity_pct", "temp_c", "traffic_index"]
THRESHOLD = 0.4

frame = pd.read_csv(ROOT / "data" / "readings.csv")
X, y = frame[FEATURES].to_numpy(), frame["exceeds_tomorrow"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

(ROOT / "model").mkdir(exist_ok=True)
joblib.dump(model, ROOT / "model" / "model.joblib")
meta = {
    "model_version": "1.0.0",
    "features": FEATURES,
    "threshold": THRESHOLD,
    "exceed_rate": round(float(y.mean()), 4),
    "test_roc_auc": round(float(auc), 4),
    "sklearn_version": sklearn.__version__,
}
(ROOT / "model" / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta))
