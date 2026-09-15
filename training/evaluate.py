"""
Stage 3 - Standalone evaluation of the serialised model.

Loads `training/model/model.pkl` and re-scores it against the same stratified held-out
split used during training, then prints a report. Use this to confirm a shipped
`.pkl` still behaves as documented without retraining it.

Run:
    python -m training.evaluate
"""

from __future__ import annotations

import json
from typing import Dict

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from app.knowledgebase.clinical_reference import FEATURE_ORDER, get_reference
from training.preprocess import load_raw, split_features_target
from training.train import (
    EVALUATION_PATH,
    METADATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
    _metrics,
)

#: Illustrative profiles used as a behavioural smoke test. These are synthetic
#: parameter sets, not real patients.
SMOKE_PROFILES: Dict[str, Dict[str, float]] = {
    "Healthy young adult": {
        "Pregnancies": 1, "Glucose": 95, "BloodPressure": 70, "SkinThickness": 22,
        "Insulin": 85, "BMI": 22.5, "DiabetesPedigreeFunction": 0.21, "Age": 26,
    },
    "Borderline midlife": {
        "Pregnancies": 3, "Glucose": 148, "BloodPressure": 84, "SkinThickness": 33,
        "Insulin": 150, "BMI": 29.4, "DiabetesPedigreeFunction": 0.52, "Age": 41,
    },
    "High-risk profile": {
        "Pregnancies": 8, "Glucose": 197, "BloodPressure": 96, "SkinThickness": 45,
        "Insulin": 320, "BMI": 38.9, "DiabetesPedigreeFunction": 1.31, "Age": 54,
    },
}


def main() -> None:
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"No model found at {MODEL_PATH}. Run `python -m training.train` first."
        )

    print("=" * 68)
    print(" DIABETES RISK MODEL - EVALUATION REPORT")
    print("=" * 68)

    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    print(f"\nModel      {metadata['model_name']} v{metadata['model_version']}")
    print(f"Trained    {metadata['trained_at']}")
    print(f"Algorithm  {metadata['algorithm']}")
    print(f"Built with scikit-learn {metadata['environment']['scikit_learn']}, "
          f"Python {metadata['environment']['python']}")

    # Reproduce the exact held-out split used at training time.
    raw = load_raw()
    features, target = split_features_target(raw)
    _, x_test, _, y_test = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=target
    )

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    metrics = _metrics(y_test.to_numpy(), y_pred, y_proba)

    print(f"\nHeld-out test set ({len(x_test)} rows)")
    print("-" * 68)
    for name in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc"):
        print(f"  {name:<12} {metrics[name]:.4f}")

    matrix = metrics["confusion_matrix"]
    print("\n  Confusion matrix")
    print("                    predicted no    predicted yes")
    print(f"    actual no    {matrix['true_negative']:>10}     {matrix['false_positive']:>12}")
    print(f"    actual yes   {matrix['false_negative']:>10}     {matrix['true_positive']:>12}")
    print(f"\n  {matrix['false_negative']} missed positive cases. In a screening tool a false "
          f"negative\n  (telling an at-risk person they are fine) is the costlier error.")

    # Confirm the report on disk still matches the live model.
    if EVALUATION_PATH.exists():
        stored = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))["test_metrics"]
        drift = [
            name for name in ("accuracy", "recall", "roc_auc")
            if abs(stored[name] - metrics[name]) > 1e-6
        ]
        verdict = "matches evaluation_report.json" if not drift else (
            f"DIVERGES from evaluation_report.json on: {', '.join(drift)}"
        )
        print(f"\n  Reproducibility check: {verdict}")

    # Behavioural smoke test across the risk spectrum.
    reference = get_reference()
    print("\nBehavioural check on synthetic profiles")
    print("-" * 68)
    for label, profile in SMOKE_PROFILES.items():
        frame = pd.DataFrame([[profile[k] for k in FEATURE_ORDER]], columns=list(FEATURE_ORDER))
        probability = float(model.predict_proba(frame)[0][1])
        band = reference.band_for_probability(probability)
        print(f"  {label:<22} {probability * 100:>5.1f}%   {band['label']}")

    print("\nEvaluation complete.\n")


if __name__ == "__main__":
    main()
