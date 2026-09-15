"""
Stage 1 - Data loading and pre-processing.

The Pima dataset encodes "not measured" as a literal 0 in five columns where a
zero is physiologically impossible (a living person cannot have a plasma glucose
of 0 mg/dL). Treating those zeros as real values drags the medians down and
teaches the model that 0 is a meaningful low reading.

They are handled with a median imputer that is fitted *inside* the model
pipeline, so the medians are learned from the training fold only and never leak
information from the test set into training.

Run standalone to produce the cleaned dataset and an audit report:
    python -m training.preprocess
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from app.knowledgebase.clinical_reference import FEATURE_ORDER

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = Path(__file__).resolve().parent
RAW_CSV = TRAINING_DIR / "data" / "raw" / "pima_indians_diabetes.csv"
PROCESSED_DIR = TRAINING_DIR / "data" / "processed"
PROCESSED_CSV = PROCESSED_DIR / "diabetes_processed.csv"
REPORT_JSON = PROCESSED_DIR / "preprocessing_report.json"

TARGET_COLUMN = "Outcome"

#: Columns where a recorded 0 means "missing", not "zero".
#: `Pregnancies` is deliberately excluded - 0 pregnancies is a valid value.
ZERO_AS_MISSING = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load the raw dataset and assert its shape before anything downstream runs."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place the Pima Indians Diabetes CSV there."
        )

    frame = pd.read_csv(path)

    expected = set(FEATURE_ORDER) | {TARGET_COLUMN}
    missing = expected - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    # Reindex to the canonical order so the model never depends on CSV column order.
    return frame[list(FEATURE_ORDER) + [TARGET_COLUMN]]


def audit_missing(frame: pd.DataFrame) -> Dict[str, Any]:
    """Count the disguised-missing zeros per column, before any imputation."""
    counts = {
        column: int((frame[column] == 0).sum()) for column in ZERO_AS_MISSING
    }
    total_rows = len(frame)
    return {
        "total_rows": total_rows,
        "zero_encoded_missing": counts,
        "zero_encoded_missing_percent": {
            column: round(count / total_rows * 100, 2) for column, count in counts.items()
        },
        "rows_with_any_missing": int((frame[list(ZERO_AS_MISSING)] == 0).any(axis=1).sum()),
        "class_balance": {
            "negative_0": int((frame[TARGET_COLUMN] == 0).sum()),
            "positive_1": int((frame[TARGET_COLUMN] == 1).sum()),
        },
    }


def build_preprocessor() -> ColumnTransformer:
    """
    The pre-processing half of the model pipeline.

    Replaces zeros with the column median in the five affected columns and
    passes the remaining columns (Pregnancies, DiabetesPedigreeFunction, Age)
    through untouched. Random Forests are scale-invariant, so no normalisation
    step is required.

    Returned unfitted - `ml.train` fits it as part of the full pipeline so the
    medians come from the training split only.
    """
    return ColumnTransformer(
        transformers=[
            (
                "impute_zeros",
                SimpleImputer(missing_values=0, strategy="median"),
                list(ZERO_AS_MISSING),
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the 8 clinical parameters from the outcome label."""
    return frame[list(FEATURE_ORDER)].copy(), frame[TARGET_COLUMN].copy()


def describe(frame: pd.DataFrame) -> Dict[str, Any]:
    """Per-column summary statistics, used on the Model Card page."""
    summary = {}
    for column in FEATURE_ORDER:
        series = frame[column]
        summary[column] = {
            "min": round(float(series.min()), 3),
            "max": round(float(series.max()), 3),
            "mean": round(float(series.mean()), 3),
            "median": round(float(series.median()), 3),
            "std": round(float(series.std()), 3),
        }
    return summary


def main() -> None:
    """Produce the cleaned dataset and a human-readable audit report."""
    print("[preprocess] Loading raw dataset ...")
    raw = load_raw()
    print(f"[preprocess] Loaded {len(raw)} rows x {raw.shape[1]} columns from {RAW_CSV.name}")

    audit = audit_missing(raw)
    total_missing = sum(audit["zero_encoded_missing"].values())
    print(f"[preprocess] Zero-encoded missing values found: {total_missing}")
    for column, count in audit["zero_encoded_missing"].items():
        pct = audit["zero_encoded_missing_percent"][column]
        state = "clean" if count == 0 else f"{count} rows ({pct}%)"
        print(f"             {column:<16} {state}")

    if total_missing == 0:
        print(
            "[preprocess] Note: this CSV has already had its zero-encoded values "
            "imputed upstream. The in-pipeline imputer stays in place so the model "
            "still handles raw Pima data correctly at inference time."
        )

    # Materialise a cleaned copy for inspection and for the deliverables ZIP.
    # This uses a whole-dataset median purely for the human-readable artefact;
    # the model itself imputes inside the pipeline on training folds only.
    features, target = split_features_target(raw)
    inspection = features.copy()
    for column in ZERO_AS_MISSING:
        median = inspection.loc[inspection[column] != 0, column].median()
        inspection[column] = inspection[column].replace(0, median)
    inspection[TARGET_COLUMN] = target

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    inspection.to_csv(PROCESSED_CSV, index=False)
    print(f"[preprocess] Wrote cleaned dataset -> {PROCESSED_CSV.relative_to(PROJECT_ROOT)}")

    report = {
        "source_file": RAW_CSV.name,
        "target_column": TARGET_COLUMN,
        "feature_order": list(FEATURE_ORDER),
        "zero_as_missing_columns": list(ZERO_AS_MISSING),
        "audit": audit,
        "summary_statistics": describe(inspection),
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[preprocess] Wrote audit report   -> {REPORT_JSON.relative_to(PROJECT_ROOT)}")
    print("[preprocess] Done.")


if __name__ == "__main__":
    main()
