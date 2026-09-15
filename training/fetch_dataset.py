"""
Stage 0 - Dataset sourcing and integrity verification.

Downloads the authentic Pima Indians Diabetes Dataset and verifies it against
the known fingerprint of the original UCI release before letting it into
`data/raw/`.

WHY THIS STAGE EXISTS
---------------------
Many copies of the Pima dataset circulating online have already had their
zero-encoded missing values "cleaned" by replacing each zero with the median of
that column *within its outcome class*. That is target leakage: it writes the
answer into the feature. In one such copy shipped with this project, Insulin
102.5 mapped to a negative outcome 100% of the time and Insulin 169.5 mapped to
a positive outcome 100% of the time, across 49% of all rows. A model trained on
it scored ROC-AUC 0.95 by reading the label rather than learning the medicine.

This stage guards against that by asserting the zeros are still present. If a
future download arrives pre-imputed, verification fails loudly instead of
silently producing an inflated score.

Run:
    python -m training.fetch_dataset
"""

from __future__ import annotations

import shutil
import ssl
import urllib.request
from pathlib import Path
from typing import Dict

import pandas as pd

from app.knowledgebase.clinical_reference import FEATURE_ORDER
from training.preprocess import RAW_CSV, TARGET_COLUMN, ZERO_AS_MISSING

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_URL = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
SOURCE_DESCRIPTION = (
    "Pima Indians Diabetes Dataset, originally from the National Institute of "
    "Diabetes and Digestive and Kidney Diseases, distributed via the UCI Machine "
    "Learning Repository."
)

#: Fingerprint of the untouched original. Any deviation means the copy has been
#: modified and must not be trusted for training.
EXPECTED_ROWS = 768
EXPECTED_CLASS_BALANCE = {"negative": 500, "positive": 268}
EXPECTED_ZERO_COUNTS: Dict[str, int] = {
    "Glucose": 5,
    "BloodPressure": 35,
    "SkinThickness": 227,
    "Insulin": 374,
    "BMI": 11,
}


class DatasetIntegrityError(RuntimeError):
    """Raised when a downloaded copy does not match the original UCI release."""


def download(url: str = SOURCE_URL, timeout: int = 30) -> str:
    print(f"[fetch] Downloading from {url}")
    context = ssl.create_default_context()
    request = urllib.request.Request(url, headers={"User-Agent": "diabetes-risk-app/1.0"})
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        return response.read().decode("utf-8")


def verify(frame: pd.DataFrame) -> None:
    """Assert the frame is the genuine, un-imputed original."""
    problems = []

    missing_columns = (set(FEATURE_ORDER) | {TARGET_COLUMN}) - set(frame.columns)
    if missing_columns:
        raise DatasetIntegrityError(
            f"Missing columns: {', '.join(sorted(missing_columns))}"
        )

    if len(frame) != EXPECTED_ROWS:
        problems.append(f"expected {EXPECTED_ROWS} rows, found {len(frame)}")

    balance = {
        "negative": int((frame[TARGET_COLUMN] == 0).sum()),
        "positive": int((frame[TARGET_COLUMN] == 1).sum()),
    }
    if balance != EXPECTED_CLASS_BALANCE:
        problems.append(f"class balance {balance} != {EXPECTED_CLASS_BALANCE}")

    print("[fetch] Verifying zero-encoded missing values are intact:")
    for column, expected in EXPECTED_ZERO_COUNTS.items():
        found = int((frame[column] == 0).sum())
        status = "ok" if found == expected else f"MISMATCH (expected {expected})"
        print(f"        {column:<16} {found:>4} zeros   {status}")
        if found != expected:
            problems.append(f"{column} has {found} zeros, expected {expected}")

    if problems:
        raise DatasetIntegrityError(
            "This copy of the dataset does not match the original UCI release:\n  - "
            + "\n  - ".join(problems)
            + "\n\nA copy with no zeros has most likely been pre-imputed. If that "
            "imputation was class-conditional it leaks the target, so it must not "
            "be used for training."
        )


def leakage_scan(frame: pd.DataFrame) -> None:
    """
    Warn if any single value in a feature column predicts the outcome perfectly.

    This is the specific signature of class-conditional imputation.
    """
    print("[fetch] Scanning for target leakage:")
    suspicious = []
    for column in ZERO_AS_MISSING:
        non_zero = frame[frame[column] != 0]
        counts = non_zero[column].value_counts()
        for value, count in counts.items():
            if count < 20:  # too rare to be an imputation artefact
                continue
            rate = non_zero.loc[non_zero[column] == value, TARGET_COLUMN].mean()
            if rate in (0.0, 1.0):
                suspicious.append((column, value, count, rate))

    if suspicious:
        for column, value, count, rate in suspicious:
            print(f"        LEAK  {column}={value} appears {count}x, "
                  f"outcome always {int(rate)}")
        raise DatasetIntegrityError(
            "Target leakage detected: a repeated feature value perfectly predicts "
            "the outcome. Do not train on this copy."
        )
    print("        No leaking constant values found.")


def main() -> None:
    print("=" * 68)
    print(" DATASET SOURCING AND INTEGRITY VERIFICATION")
    print("=" * 68)
    print(f"\n{SOURCE_DESCRIPTION}\n")

    raw_text = download()
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)

    temporary = RAW_CSV.with_suffix(".download.tmp")
    temporary.write_text(raw_text, encoding="utf-8")

    try:
        frame = pd.read_csv(temporary)
        verify(frame)
        leakage_scan(frame)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    # Only overwrite the trusted path once every check has passed.
    shutil.move(str(temporary), str(RAW_CSV))
    print(f"\n[fetch] Verified. Wrote {len(frame)} rows -> "
          f"{RAW_CSV.relative_to(PROJECT_ROOT)}")
    print("[fetch] Next: python -m training.preprocess\n")


if __name__ == "__main__":
    main()
