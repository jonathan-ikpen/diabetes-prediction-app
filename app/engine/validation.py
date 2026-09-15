"""
Input validation and sanitisation for the 8 clinical parameters.

Every value entering the model passes through here first. The accepted range for
each parameter is read from the knowledge base rather than hardcoded, so the
clinical thresholds and the input bounds can never drift apart.

The rules are deliberately strict: anything that is not a finite number inside
the documented range is rejected with a message a non-technical user can act on.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Mapping, Tuple

from app.knowledgebase.clinical_reference import FEATURE_ORDER, get_reference

#: Parameters recorded as whole numbers. A fractional count of pregnancies is
#: a data-entry error, not a measurement.
INTEGER_FIELDS = frozenset({"Pregnancies", "Age"})

#: Longest string accepted for any single field, before parsing. Bounds the work
#: done on hostile input.
MAX_INPUT_LENGTH = 24

#: A plain decimal number, optionally signed. Deliberately rejects hex, infinity,
#: nan, exponent notation, and anything else Python's float() would happily
#: accept but a clinician would never type.
_NUMBER_PATTERN = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)$")


class ValidationError(Exception):
    """Raised when a submission cannot be sanitised into a usable parameter set.

    `errors` maps a field name to a human-readable message. Field names match
    the form input names, so the UI can attach each message to its own input.
    """

    def __init__(self, errors: Dict[str, str]) -> None:
        self.errors = errors
        super().__init__("; ".join(f"{field}: {message}" for field, message in errors.items()))


def _coerce_number(raw: Any, field: str) -> float:
    """Turn one submitted value into a finite float, or raise ValueError."""
    if raw is None:
        raise ValueError("This measurement is required.")

    text = str(raw).strip()
    if not text:
        raise ValueError("This measurement is required.")
    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError("This value is too long to be a valid measurement.")

    # Accept a comma as a decimal separator, which is common data entry practice.
    text = text.replace(",", ".")

    if not _NUMBER_PATTERN.match(text):
        raise ValueError("Enter a number, using digits only.")

    try:
        value = float(text)
    except (TypeError, ValueError):
        raise ValueError("Enter a number, using digits only.") from None

    if not math.isfinite(value):
        raise ValueError("Enter a real number.")

    if field in INTEGER_FIELDS and not float(value).is_integer():
        raise ValueError("Enter a whole number.")

    return value


def validate_submission(payload: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Validate and sanitise a full 8-parameter submission.

    Returns:
        (cleaned, errors). `cleaned` holds every field that parsed successfully,
        keyed in FEATURE_ORDER; `errors` holds a message per rejected field.
        Callers should treat a non-empty `errors` as a failed submission.

    Unknown keys in `payload` are ignored rather than rejected, so the model can
    never be fed a column it was not trained on.
    """
    reference = get_reference()
    cleaned: Dict[str, float] = {}
    errors: Dict[str, str] = {}

    for field in FEATURE_ORDER:
        spec = reference.parameter(field)
        low, high = reference.input_range(field)

        try:
            value = _coerce_number(payload.get(field), field)
        except ValueError as exc:
            errors[field] = str(exc)
            continue

        if value < low or value > high:
            unit = spec["unit"]
            unit_suffix = "" if unit == "count" else f" {unit}"
            errors[field] = (
                f"Enter a value between {low:g} and {high:g}{unit_suffix}. "
                f"Values outside this range are not physiologically plausible."
            )
            continue

        cleaned[field] = int(value) if field in INTEGER_FIELDS else round(value, 4)

    return cleaned, errors


def validate_or_raise(payload: Mapping[str, Any]) -> Dict[str, float]:
    """Strict variant for the JSON API, where partial results are not useful."""
    cleaned, errors = validate_submission(payload)
    if errors:
        raise ValidationError(errors)
    return cleaned
