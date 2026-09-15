"""
The plain-language interview.

The clinical assessment asks for all eight parameters directly, which suits a
health worker holding lab results. A member of the public cannot answer three
of them - serum insulin, triceps skinfold, and often plasma glucose are
laboratory measurements, not things anyone knows offhand.

This module asks only what a person can actually answer, then derives the eight
model inputs from those answers. Anything genuinely unknown is passed to the
pipeline as "not measured", where the same median imputer that handles the gaps
in the original dataset fills it in.

WHY THAT IS DEFENSIBLE
----------------------
Measured on the held-out test set, dropping insulin and skinfold costs almost
nothing, because the model barely leans on them and 49% / 30% of the training
rows were missing them anyway:

    all eight known                     ROC-AUC 0.809, sensitivity 78%
    insulin + skinfold unknown          ROC-AUC 0.807, sensitivity 78%
    those two plus glucose unknown      ROC-AUC 0.779, sensitivity 69%

Glucose is the one that matters, so the interview asks for it if the person has
ever had a blood test, and tells them plainly what it is worth.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

#: Parameters no member of the public can supply. Always sent as "not measured".
ALWAYS_UNKNOWN = ("Insulin", "SkinThickness")

#: Family history mapped onto the Diabetes Pedigree Function.
#:
#: DPF is a synthesised score weighting relatives by closeness and by how young
#: they were diagnosed, so it cannot be asked directly. These values are the
#: quartile landmarks of the real distribution in the training data (25th 0.244,
#: 50th 0.372, 75th 0.626, 90th 0.879), where the positive-outcome rate climbs
#: from 25.5% in the lowest quartile to 48.4% in the highest. The mapping is an
#: approximation, and the result page says so.
PEDIGREE_BY_FAMILY_HISTORY = {
    "none": 0.17,
    "one": 0.37,
    "several": 0.63,
    "several_young": 0.88,
}

#: The interview script. `depends_on` hides a question until an earlier answer
#: makes it relevant, which is what keeps the run short for most people.
QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "age",
        "type": "number",
        "question": "How old are you?",
        "help": "Diabetes becomes more common with age, so this is one of the strongest things we can ask.",
        "min": 18, "max": 100, "step": "1", "unit": "years",
    },
    {
        "id": "height_cm",
        "type": "number",
        "question": "How tall are you?",
        "help": "In centimetres. If you know your height in feet and inches, 5 ft 6 in is about 168 cm.",
        "min": 120, "max": 220, "step": "1", "unit": "cm",
    },
    {
        "id": "weight_kg",
        "type": "number",
        "question": "How much do you weigh?",
        "help": "In kilograms. An approximate figure is fine.",
        "min": 30, "max": 250, "step": "0.1", "unit": "kg",
    },
    {
        "id": "been_pregnant",
        "type": "choice",
        "question": "Have you ever been pregnant?",
        "help": "Pregnancy places extra demand on the body's insulin, so it affects long-term risk.",
        "options": [
            {"value": "yes", "label": "Yes"},
            {"value": "no", "label": "No"},
        ],
    },
    {
        "id": "pregnancies",
        "type": "number",
        "question": "How many times have you been pregnant?",
        "help": "Include all pregnancies, however they ended.",
        "min": 1, "max": 20, "step": "1", "unit": "times",
        "depends_on": {"id": "been_pregnant", "value": "yes"},
    },
    {
        "id": "family_history",
        "type": "choice",
        "question": "Has anyone in your family had diabetes?",
        "help": "Close relatives matter most: parents, brothers and sisters.",
        "options": [
            {"value": "none", "label": "No one that I know of"},
            {"value": "one", "label": "One parent, brother or sister"},
            {"value": "several", "label": "Several close relatives"},
            {"value": "several_young", "label": "Several, and some were diagnosed young"},
        ],
    },
    {
        "id": "knows_bp",
        "type": "choice",
        "question": "Do you know your blood pressure?",
        "help": "It is usually written as two numbers, like 120 over 80. If you have never had it measured, that is fine.",
        "options": [
            {"value": "yes", "label": "Yes, I know it"},
            {"value": "no", "label": "No, I don't"},
        ],
    },
    {
        "id": "blood_pressure",
        "type": "number",
        "question": "What is the lower of the two numbers?",
        "help": "In a reading like 120 over 80, the lower number is 80.",
        "min": 30, "max": 140, "step": "1", "unit": "mm Hg",
        "depends_on": {"id": "knows_bp", "value": "yes"},
    },
    {
        "id": "had_glucose_test",
        "type": "choice",
        "question": "Have you ever had your blood sugar measured?",
        "help": "This is the single most useful thing you can tell us. Without it the check still works, but it is less accurate.",
        "options": [
            {"value": "yes", "label": "Yes, and I know the number"},
            {"value": "no", "label": "No, or I don't remember"},
        ],
    },
    {
        "id": "glucose",
        "type": "number",
        "question": "What was the reading?",
        "help": "In mg/dL. A normal result is usually under 140. If your result was in mmol/L, multiply it by 18.",
        "min": 40, "max": 300, "step": "1", "unit": "mg/dL",
        "depends_on": {"id": "had_glucose_test", "value": "yes"},
    },
]


class InterviewError(ValueError):
    """Raised when the answers cannot be turned into model inputs."""


def _number(answers: Dict[str, Any], key: str, spec: Dict[str, Any]) -> float:
    raw = str(answers.get(key, "")).strip().replace(",", ".")
    if not raw:
        raise InterviewError(f"Please answer: {spec['question']}")
    try:
        value = float(raw)
    except ValueError:
        raise InterviewError(f"'{raw}' is not a number. {spec['question']}") from None
    if not (spec["min"] <= value <= spec["max"]):
        raise InterviewError(
            f"{spec['question']} Please give a value between {spec['min']} and {spec['max']}."
        )
    return value


def _spec(question_id: str) -> Dict[str, Any]:
    return next(q for q in QUESTIONS if q["id"] == question_id)


def derive_parameters(answers: Dict[str, Any]) -> Tuple[Dict[str, float], List[str]]:
    """
    Turn interview answers into the eight model inputs.

    Returns:
        (values, unknown) - `values` holds all eight keys; `unknown` names the
        ones that were never measured, so the caller can tell the pipeline to
        impute them and can avoid presenting them as if they were reported.
    """
    unknown: List[str] = list(ALWAYS_UNKNOWN)
    values: Dict[str, float] = {}

    values["Age"] = _number(answers, "age", _spec("age"))

    height_cm = _number(answers, "height_cm", _spec("height_cm"))
    weight_kg = _number(answers, "weight_kg", _spec("weight_kg"))
    bmi = weight_kg / (height_cm / 100) ** 2
    if not (12 <= bmi <= 70):
        raise InterviewError(
            f"That height and weight give a body mass index of {bmi:.1f}, which is outside "
            "the range this tool can score. Please check both numbers."
        )
    values["BMI"] = round(bmi, 2)

    if str(answers.get("been_pregnant", "")).lower() == "yes":
        values["Pregnancies"] = _number(answers, "pregnancies", _spec("pregnancies"))
    else:
        values["Pregnancies"] = 0.0

    history = str(answers.get("family_history", "")).lower()
    if history not in PEDIGREE_BY_FAMILY_HISTORY:
        raise InterviewError("Please answer the question about family history.")
    values["DiabetesPedigreeFunction"] = PEDIGREE_BY_FAMILY_HISTORY[history]

    if str(answers.get("knows_bp", "")).lower() == "yes":
        values["BloodPressure"] = _number(answers, "blood_pressure", _spec("blood_pressure"))
    else:
        values["BloodPressure"] = 0.0
        unknown.append("BloodPressure")

    if str(answers.get("had_glucose_test", "")).lower() == "yes":
        values["Glucose"] = _number(answers, "glucose", _spec("glucose"))
    else:
        values["Glucose"] = 0.0
        unknown.append("Glucose")

    for key in ALWAYS_UNKNOWN:
        values[key] = 0.0

    return values, unknown


def accuracy_note(unknown: List[str]) -> str:
    """A plain-language line about how much the missing answers cost."""
    if "Glucose" in unknown:
        return (
            "This estimate was made without a blood sugar reading, which is the single "
            "most useful measurement. Tested on records the model had never seen, leaving "
            "it out lowers the share of true cases caught from about 78% to about 69%. "
            "Treat this result as a rough guide."
        )
    return (
        "Serum insulin and skinfold thickness were not measured, and were estimated from "
        "typical values. On the held-out test set that makes almost no difference to "
        "accuracy, because the model barely relies on them."
    )
