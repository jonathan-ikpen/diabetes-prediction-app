"""
Stage 2 - Model training, evaluation, and serialisation.

Trains a Random Forest classifier on the 8 clinical parameters and writes three
artefacts into /model:

    model.pkl               the fitted Pipeline (imputer + classifier)
    model_metadata.json     what was trained, on what, with which settings
    evaluation_report.json  held-out metrics, confusion matrix, ROC curve,
                            and feature importances (rendered on /model-card)

The serialised object is the *whole pipeline*, not the bare classifier, so
inference applies exactly the same imputation that training used.

Run:
    python -m training.train
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from app.knowledgebase.clinical_reference import FEATURE_ORDER, get_reference
from training.preprocess import (
    PROJECT_ROOT,
    TRAINING_DIR,
    TARGET_COLUMN,
    ZERO_AS_MISSING,
    audit_missing,
    build_preprocessor,
    describe,
    load_raw,
    split_features_target,
)

MODEL_DIR = TRAINING_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
EVALUATION_PATH = MODEL_DIR / "evaluation_report.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

#: Minimum sensitivity (recall) the screening threshold must achieve.
#:
#: At the textbook 0.5 cut-off this model catches only ~54% of positive cases.
#: For a screening instrument that is the wrong trade: telling an at-risk person
#: they are fine (a false negative) costs far more than sending a healthy person
#: for a confirmatory blood test (a false positive). The operating threshold is
#: therefore lowered until 80% of true cases are caught, and the cost is paid in
#: specificity. The threshold is chosen on out-of-fold TRAINING predictions only,
#: so the held-out test set stays untouched.
TARGET_SENSITIVITY = 0.80

#: Searched with 5-fold stratified cross-validation, scored on ROC-AUC.
PARAM_GRID: Dict[str, List[Any]] = {
    "classifier__n_estimators": [200, 400],
    "classifier__max_depth": [None, 6, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": ["sqrt", 0.5],
    # Positives are ~35% of the cohort. Screening tools favour sensitivity, so
    # the balanced weighting is offered to the search rather than assumed.
    "classifier__class_weight": ["balanced", None],
}


def build_pipeline() -> Pipeline:
    """Imputation and classification as one serialisable unit."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            ),
        ]
    )


def _feature_importances(pipeline: Pipeline) -> List[Dict[str, Any]]:
    """Map the fitted forest's importances back onto readable parameter names."""
    reference = get_reference()
    names = list(pipeline.named_steps["preprocess"].get_feature_names_out())
    importances = pipeline.named_steps["classifier"].feature_importances_

    rows = []
    for name, importance in zip(names, importances):
        rows.append(
            {
                "feature": name,
                "display_name": reference.parameter(name)["display_name"],
                "unit": reference.parameter(name)["unit"],
                "importance": round(float(importance), 5),
            }
        )
    return sorted(rows, key=lambda row: row["importance"], reverse=True)


def _roc_points(y_true: np.ndarray, y_score: np.ndarray, max_points: int = 60) -> List[Dict[str, float]]:
    """ROC curve, thinned to a size that draws cleanly as an inline SVG."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    if len(fpr) > max_points:
        index = np.linspace(0, len(fpr) - 1, max_points).astype(int)
        fpr, tpr = fpr[index], tpr[index]
    return [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in zip(fpr, tpr)]


def choose_screening_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, target_sensitivity: float = TARGET_SENSITIVITY
) -> float:
    """
    Pick the highest probability cut-off that still catches `target_sensitivity`
    of true positive cases.

    Taking the *highest* qualifying threshold keeps specificity as high as the
    sensitivity requirement allows, so the tool refers as few healthy people for
    unnecessary testing as possible while still meeting its screening duty.
    """
    candidates = np.unique(np.round(y_proba, 3))
    qualifying = [
        threshold
        for threshold in candidates
        if recall_score(y_true, (y_proba >= threshold).astype(int), zero_division=0)
        >= target_sensitivity
    ]
    if not qualifying:
        return 0.5
    return float(max(qualifying))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(specificity), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def main() -> None:
    print("=" * 68)
    print(" DIABETES RISK MODEL - TRAINING PIPELINE")
    print("=" * 68)

    # ---------------------------------------------------------------- 1. data
    raw = load_raw()
    features, target = split_features_target(raw)
    audit = audit_missing(raw)
    print(f"\n[1/5] Dataset: {len(raw)} rows, {len(FEATURE_ORDER)} features")
    print(f"      Class balance: {audit['class_balance']['negative_0']} negative / "
          f"{audit['class_balance']['positive_1']} positive")

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,  # preserve the 65/35 class ratio in both splits
    )
    print(f"      Split: {len(x_train)} train / {len(x_test)} held-out test "
          f"(stratified, seed={RANDOM_STATE})")

    # ------------------------------------------------------------ 2. search
    print(f"\n[2/5] Grid search over {np.prod([len(v) for v in PARAM_GRID.values()])} "
          f"configurations, {CV_FOLDS}-fold stratified CV, scored on ROC-AUC ...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)
    model: Pipeline = search.best_estimator_

    best_params = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
    print(f"      Best cross-validated ROC-AUC: {search.best_score_:.4f}")
    for key, value in best_params.items():
        print(f"      {key:<20} {value}")

    # -------------------------------------------------------- 3. cv stability
    print(f"\n[3/5] Re-running {CV_FOLDS}-fold CV on the winning configuration ...")
    cv_scores = cross_val_score(model, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    print(f"      ROC-AUC per fold: {', '.join(f'{s:.4f}' for s in cv_scores)}")
    print(f"      Mean {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # --------------------------------------------------- 3b. screening cutoff
    print(f"\n[4/6] Selecting a screening threshold for >= {TARGET_SENSITIVITY:.0%} "
          f"sensitivity, using out-of-fold training predictions ...")
    oof_proba = cross_val_predict(
        model, x_train, y_train, cv=cv, method="predict_proba", n_jobs=-1
    )[:, 1]
    screening_threshold = choose_screening_threshold(y_train.to_numpy(), oof_proba)
    oof_recall = recall_score(y_train, (oof_proba >= screening_threshold).astype(int))
    print(f"      Threshold {screening_threshold:.3f} (down from the default 0.500)")
    print(f"      Out-of-fold sensitivity at this threshold: {oof_recall:.4f}")

    # ------------------------------------------------------- 4. held-out eval
    print("\n[5/6] Evaluating on the held-out test set ...")
    y_proba = model.predict_proba(x_test)[:, 1]
    y_pred = model.predict(x_test)                                  # default 0.5
    y_pred_screening = (y_proba >= screening_threshold).astype(int)  # tuned cut-off

    test_metrics = _metrics(y_test.to_numpy(), y_pred, y_proba)
    screening_metrics = _metrics(y_test.to_numpy(), y_pred_screening, y_proba)

    train_proba = model.predict_proba(x_train)[:, 1]
    train_metrics = _metrics(y_train.to_numpy(), model.predict(x_train), train_proba)

    print(f"\n      {'metric':<14}{'@ 0.500 default':>18}{'@ ' + f'{screening_threshold:.3f} screening':>22}")
    print(f"      {'-' * 52}")
    for name in ("accuracy", "precision", "recall", "specificity", "f1"):
        marker = "  <-- sensitivity" if name == "recall" else ""
        print(f"      {name:<14}{test_metrics[name]:>18.4f}{screening_metrics[name]:>22.4f}{marker}")
    print(f"      {'roc_auc':<14}{test_metrics['roc_auc']:>18.4f}"
          f"{'(threshold-free)':>22}")

    for label, metrics in (("default 0.500", test_metrics), (f"screening {screening_threshold:.3f}", screening_metrics)):
        matrix = metrics["confusion_matrix"]
        print(f"\n      Confusion @ {label}:  TN={matrix['true_negative']} "
              f"FP={matrix['false_positive']} FN={matrix['false_negative']} "
              f"TP={matrix['true_positive']}")
        print(f"        -> {matrix['false_negative']} at-risk people missed, "
              f"{matrix['false_positive']} healthy people sent for confirmatory testing")

    gap = train_metrics["roc_auc"] - test_metrics["roc_auc"]
    print(f"      Train-test ROC-AUC gap: {gap:.4f} "
          f"({'acceptable' if gap < 0.15 else 'REVIEW - possible overfitting'})")

    importances = _feature_importances(model)
    print("\n      Feature importance ranking:")
    for row in importances:
        bar = "#" * max(1, int(row["importance"] * 60))
        print(f"      {row['display_name']:<28} {row['importance']:.4f}  {bar}")

    # ----------------------------------------------------------- 5. serialise
    print("\n[6/6] Serialising artefacts ...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)

    metadata = {
        "model_name": "Diabetes Risk Random Forest",
        "model_version": "1.0.0",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": "RandomForestClassifier (scikit-learn)",
        "pipeline_steps": ["SimpleImputer(missing_values=0, strategy=median)", "RandomForestClassifier"],
        "feature_order": list(FEATURE_ORDER),
        "target_column": TARGET_COLUMN,
        "zero_as_missing_columns": list(ZERO_AS_MISSING),
        "positive_class_meaning": "Tested positive for diabetes in the source cohort",
        "decision_threshold": 0.5,
        "screening_threshold": round(screening_threshold, 3),
        "target_sensitivity": TARGET_SENSITIVITY,
        "threshold_note": (
            "The app reports a calibrated probability and a named risk band rather "
            "than a hard label. `screening_threshold` is the cut-off at which the "
            "model meets its target sensitivity, selected on out-of-fold training "
            "predictions only."
        ),
        "hyperparameters": {k: (v if v is not None else "None") for k, v in best_params.items()},
        "training": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "search_scoring": "roc_auc",
            "n_train_rows": int(len(x_train)),
            "n_test_rows": int(len(x_test)),
        },
        "dataset": {
            "name": "Pima Indians Diabetes Dataset",
            "rows": int(len(raw)),
            "features": len(FEATURE_ORDER),
            "class_balance": audit["class_balance"],
            "summary_statistics": describe(raw),
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    evaluation = {
        "generated_at": metadata["trained_at"],
        "test_metrics": test_metrics,
        "screening_metrics": screening_metrics,
        "screening_threshold": round(screening_threshold, 3),
        "target_sensitivity": TARGET_SENSITIVITY,
        "train_metrics": train_metrics,
        "overfitting_gap_roc_auc": round(float(gap), 4),
        "cross_validation": {
            "folds": CV_FOLDS,
            "scoring": "roc_auc",
            "scores": [round(float(s), 4) for s in cv_scores],
            "mean": round(float(cv_scores.mean()), 4),
            "std": round(float(cv_scores.std()), 4),
        },
        "best_cv_score": round(float(search.best_score_), 4),
        "feature_importances": importances,
        "roc_curve": _roc_points(y_test.to_numpy(), y_proba),
    }
    EVALUATION_PATH.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"      {MODEL_PATH.relative_to(PROJECT_ROOT)}  ({size_kb:.0f} KB)")
    print(f"      {METADATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"      {EVALUATION_PATH.relative_to(PROJECT_ROOT)}")
    print("\nTraining complete.\n")


if __name__ == "__main__":
    main()
