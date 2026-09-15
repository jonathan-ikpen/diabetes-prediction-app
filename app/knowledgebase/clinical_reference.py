"""
Read-only accessor over `who_idf_thresholds.json`.

This module is the ONLY place allowed to interpret the threshold file. Every
other layer (inference engine, Flask views, templates) asks this module rather
than hardcoding a cut-off, so a change to the JSON propagates everywhere.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_THRESHOLDS_PATH = Path(__file__).resolve().parent / "who_idf_thresholds.json"

#: Canonical feature order. The trained model expects columns in exactly this
#: order, so this tuple is the single source of truth for the whole project.
FEATURE_ORDER = (
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
)

#: Severity ranking, used to sort findings so the most clinically urgent
#: parameters are presented first.
_SEVERITY_RANK = {"critical": 0, "watch": 1, "normal": 2}


@dataclass(frozen=True)
class ParameterFinding:
    """The clinical interpretation of one submitted parameter value."""

    key: str
    display_name: str
    clinical_name: str
    value: float
    unit: str
    band_id: str
    band_label: str
    severity: str          # "critical" | "watch" | "normal"
    note: str
    guideline: str
    source: str
    is_diagnostic: bool

    @property
    def is_flagged(self) -> bool:
        """True when the value sits outside the normal band."""
        return self.severity != "normal"

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["is_flagged"] = self.is_flagged
        return data


class ClinicalReference:
    """Loads the WHO/IDF threshold file and answers clinical questions about values."""

    def __init__(self, path: Path | str = _THRESHOLDS_PATH) -> None:
        self._path = Path(path)
        with self._path.open("r", encoding="utf-8") as handle:
            self._data: Dict[str, Any] = json.load(handle)
        self._validate()

    # ---------------------------------------------------------------- loading

    def _validate(self) -> None:
        """Fail loudly at import time rather than silently mis-classifying later."""
        missing = [k for k in FEATURE_ORDER if k not in self._data.get("parameters", {})]
        if missing:
            raise ValueError(
                f"{self._path.name} is missing threshold definitions for: {', '.join(missing)}"
            )
        if not self._data.get("risk_bands"):
            raise ValueError(f"{self._path.name} defines no risk_bands.")

    # -------------------------------------------------------------- accessors

    @property
    def raw(self) -> Dict[str, Any]:
        """The whole document, for pages that render the knowledge base directly."""
        return self._data

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._data["parameters"]

    @property
    def risk_bands(self) -> List[Dict[str, Any]]:
        return self._data["risk_bands"]

    @property
    def general_guidance(self) -> List[Dict[str, Any]]:
        return self._data["general_guidance"]

    @property
    def regional_context(self) -> Dict[str, Any]:
        return self._data["regional_context"]

    @property
    def disclaimer(self) -> str:
        return self._data["disclaimer"]

    def parameter(self, key: str) -> Dict[str, Any]:
        try:
            return self._data["parameters"][key]
        except KeyError as exc:  # pragma: no cover - guarded by _validate
            raise KeyError(f"Unknown clinical parameter: {key!r}") from exc

    def input_range(self, key: str) -> tuple[float, float]:
        """Accepted (min, max) for a parameter, used by the validation layer."""
        low, high = self.parameter(key)["input_range"]
        return float(low), float(high)

    # ---------------------------------------------------------- interpretation

    def classify_value(self, key: str, value: float) -> ParameterFinding:
        """Place a single value into its WHO/IDF band."""
        spec = self.parameter(key)
        band = self._match_band(spec["bands"], float(value))
        return ParameterFinding(
            key=key,
            display_name=spec["display_name"],
            clinical_name=spec["clinical_name"],
            value=float(value),
            unit=spec["unit"],
            band_id=band["id"],
            band_label=band["label"],
            severity=band["severity"],
            note=band["note"],
            guideline=spec["guideline"],
            source=spec["source"],
            is_diagnostic=bool(spec.get("is_diagnostic", False)),
        )

    @staticmethod
    def _match_band(bands: List[Dict[str, Any]], value: float) -> Dict[str, Any]:
        """
        Return the first band whose [min, max) interval contains `value`.

        Bands use half-open intervals so adjacent bands (e.g. BMI 25-30 and
        30-35) never both match a boundary value such as exactly 30.0.
        """
        for band in bands:
            low = band.get("min")
            high = band.get("max")
            if low is not None and value < low:
                continue
            if high is not None and value >= high:
                continue
            return band
        # Defensive: the JSON is authored so this is unreachable, but a value
        # outside every band should degrade rather than crash a screening.
        return bands[-1]

    @staticmethod
    def sort_findings(findings: List[ParameterFinding]) -> List[ParameterFinding]:
        """Most clinically urgent first, ties broken by FEATURE_ORDER."""
        return sorted(
            findings,
            key=lambda f: (_SEVERITY_RANK.get(f.severity, 3), FEATURE_ORDER.index(f.key)),
        )

    def classify_all(self, values: Dict[str, float]) -> List[ParameterFinding]:
        """Interpret a full submission, most clinically urgent parameter first."""
        return self.sort_findings(
            [self.classify_value(key, values[key]) for key in FEATURE_ORDER]
        )

    def band_for_probability(self, probability: float) -> Dict[str, Any]:
        """Map a model probability (0.0-1.0) onto a named risk band."""
        for band in self.risk_bands:
            if band["min_probability"] <= probability < band["max_probability"]:
                return band
        return self.risk_bands[-1]


def rank_by_influence(
    findings: List[ParameterFinding],
    top_drivers: Optional[List[Dict[str, Any]]],
) -> List[ParameterFinding]:
    """
    Reorder findings by how much the model actually relies on each parameter.

    `classify_all` orders by clinical severity, which is the right order to
    *show* a patient. For explaining *why the model produced this score*, the
    model's own feature weighting is the honest order - otherwise the narrative
    can lead with a parameter the forest barely uses.

    Falls back to the given order when no importances are available.
    """
    if not top_drivers:
        return list(findings)
    rank = {driver["feature"]: index for index, driver in enumerate(top_drivers)}
    return sorted(findings, key=lambda f: rank.get(f.key, len(rank)))


_reference: Optional[ClinicalReference] = None


def get_reference() -> ClinicalReference:
    """Process-wide singleton. The threshold file is read once per process."""
    global _reference
    if _reference is None:
        _reference = ClinicalReference()
    return _reference
