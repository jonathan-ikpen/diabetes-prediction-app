"""
The mathematical half of the hybrid architecture.

Loads `training/model/model.pkl` exactly once per process and turns a validated
parameter set into a probability, a named risk band, and the WHO/IDF reading of
each submitted value.

No text generation happens here. This layer is fully deterministic: the same
input always produces the same output, with or without an LLM API key.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd

from app.knowledgebase.clinical_reference import (
    FEATURE_ORDER,
    ParameterFinding,
    get_reference,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# The trained model is an output of the training pipeline, so it lives with it.
MODEL_DIR = PROJECT_ROOT / "training" / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
EVALUATION_PATH = MODEL_DIR / "evaluation_report.json"


class ModelNotTrainedError(RuntimeError):
    """Raised when the app starts without a serialised model on disk."""


@dataclass
class PredictionResult:
    """Everything the deterministic layer knows about one screening."""

    probability: float                     # 0.0 - 1.0, positive class
    percentage: float                      # probability as 0.0 - 100.0, 1 dp
    band_id: str
    band_label: str
    band_summary: str
    band_action: str
    exceeds_screening_threshold: bool
    screening_threshold: float
    findings: List[ParameterFinding] = field(default_factory=list)
    top_drivers: List[Dict[str, Any]] = field(default_factory=list)
    inputs: Dict[str, float] = field(default_factory=dict)
    #: Parameters that were never measured. The pipeline imputed them, and they
    #: are deliberately absent from `findings` so the interface never shows an
    #: estimated value as though it had been reported.
    unknown: List[str] = field(default_factory=list)

    @property
    def flagged_findings(self) -> List[ParameterFinding]:
        """Only the parameters sitting outside their normal band."""
        return [finding for finding in self.findings if finding.is_flagged]

    @property
    def critical_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "critical")

    @property
    def watch_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "watch")

    @property
    def normal_count(self) -> int:
        return sum(1 for finding in self.findings if finding.severity == "normal")

    def as_dict(self) -> Dict[str, Any]:
        """JSON-safe representation, used by the /api/predict endpoint."""
        return {
            "probability": round(self.probability, 4),
            "percentage": self.percentage,
            "risk_band": {
                "id": self.band_id,
                "label": self.band_label,
                "summary": self.band_summary,
                "recommended_action": self.band_action,
            },
            "exceeds_screening_threshold": self.exceeds_screening_threshold,
            "screening_threshold": self.screening_threshold,
            "inputs": self.inputs,
            "unknown_parameters": self.unknown,
            "findings": [finding.as_dict() for finding in self.findings],
            "summary_counts": {
                "critical": self.critical_count,
                "watch": self.watch_count,
                "normal": self.normal_count,
            },
        }


class Predictor:
    """Thin, thread-safe wrapper around the serialised scikit-learn pipeline."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self._model_path = model_path
        self._model: Optional[Any] = None
        self._lock = threading.Lock()
        self.metadata: Dict[str, Any] = self._read_json(METADATA_PATH)
        self.evaluation: Dict[str, Any] = self._read_json(EVALUATION_PATH)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    # -------------------------------------------------------------- lifecycle

    @property
    def is_ready(self) -> bool:
        return self._model_path.exists()

    def load(self) -> Any:
        """Deserialise the pipeline on first use, then reuse it for the process.

        Guarded by a lock because Flask's threaded development server can serve
        two requests concurrently during a cold start.
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    if not self._model_path.exists():
                        raise ModelNotTrainedError(
                            f"No trained model at {self._model_path}. "
                            "Run `python -m training.train` before starting the app."
                        )
                    model = joblib.load(self._model_path)

                    # Training used n_jobs=-1, which is right for fitting hundreds
                    # of trees. At inference the app scores one row at a time, where
                    # spinning up a thread pool costs more than it saves and the
                    # threaded reduction makes the summed probability vary in the
                    # last few decimal places between identical calls. Scoring on a
                    # single thread is faster here and bit-for-bit reproducible.
                    classifier = model.named_steps.get("classifier")
                    if classifier is not None and hasattr(classifier, "n_jobs"):
                        classifier.n_jobs = 1

                    self._model = model
        return self._model

    # -------------------------------------------------------------- inference

    @property
    def screening_threshold(self) -> float:
        return float(self.metadata.get("screening_threshold", 0.5))

    @property
    def feature_importances(self) -> List[Dict[str, Any]]:
        """
        Importance rows, with their labels re-resolved from the knowledge base.

        `evaluation_report.json` stores a display name and unit alongside each
        importance so the file reads standalone, but those are presentation data
        that can be edited in the knowledge base without retraining. Re-resolving
        them here means the UI can never show a label that has since changed.
        """
        reference = get_reference()
        rows = []
        for row in self.evaluation.get("feature_importances", []):
            spec = reference.parameters.get(row["feature"])
            rows.append({
                **row,
                "display_name": spec["display_name"] if spec else row.get("display_name"),
                "unit": spec["unit"] if spec else row.get("unit"),
            })
        return rows

    def _drivers_for(self, findings: List[ParameterFinding]) -> List[Dict[str, Any]]:
        """
        Rank this person's flagged parameters by how much the model relies on them.

        Global feature importance says what the forest weights across the whole
        cohort; intersecting it with the parameters that are actually abnormal for
        this individual gives a per-person explanation rather than a generic one.
        """
        importance_by_feature = {
            row["feature"]: row["importance"] for row in self.feature_importances
        }
        flagged = [finding for finding in findings if finding.is_flagged]
        pool = flagged if flagged else findings

        drivers = [
            {
                "feature": finding.key,
                "display_name": finding.display_name,
                "value": finding.value,
                "unit": finding.unit,
                "band_label": finding.band_label,
                "severity": finding.severity,
                "importance": importance_by_feature.get(finding.key, 0.0),
            }
            for finding in pool
        ]
        return sorted(drivers, key=lambda row: row["importance"], reverse=True)

    def predict(
        self,
        values: Dict[str, float],
        unknown: Optional[List[str]] = None,
    ) -> PredictionResult:
        """
        Score one validated submission.

        Args:
            values: Already validated by `engine.validation`. Must contain every
                key in FEATURE_ORDER.
            unknown: Parameters that were never actually measured. They are sent
                to the pipeline as 0, which is how this dataset encodes "not
                measured" and what the in-pipeline median imputer expects, and
                they are left out of the clinical findings entirely - reading a
                WHO band off an imputed number would be inventing a result.
        """
        missing = [key for key in FEATURE_ORDER if key not in values]
        if missing:
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")

        unmeasured = list(unknown or [])
        model = self.load()

        # Build the frame with explicit column names in the canonical order, so
        # the pipeline's ColumnTransformer resolves columns by name rather than
        # relying on positional order.
        frame = pd.DataFrame(
            [[0.0 if key in unmeasured else values[key] for key in FEATURE_ORDER]],
            columns=list(FEATURE_ORDER),
        )
        probability = float(model.predict_proba(frame)[0][1])

        reference = get_reference()
        band = reference.band_for_probability(probability)
        findings = [
            reference.classify_value(key, values[key])
            for key in FEATURE_ORDER
            if key not in unmeasured
        ]
        findings = reference.sort_findings(findings)

        return PredictionResult(
            probability=probability,
            percentage=round(probability * 100, 1),
            band_id=band["id"],
            band_label=band["label"],
            band_summary=band["summary"],
            band_action=band["action"],
            exceeds_screening_threshold=probability >= self.screening_threshold,
            screening_threshold=self.screening_threshold,
            findings=findings,
            top_drivers=self._drivers_for(findings),
            inputs={k: v for k, v in values.items() if k not in unmeasured},
            unknown=unmeasured,
        )


_predictor: Optional[Predictor] = None
_predictor_lock = threading.Lock()


def get_predictor() -> Predictor:
    """Process-wide singleton, so `model.pkl` is read from disk only once."""
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                _predictor = Predictor()
    return _predictor
