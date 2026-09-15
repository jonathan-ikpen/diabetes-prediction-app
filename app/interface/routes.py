"""
Page controllers.

Each view validates input, delegates to the inference engine, and renders a
template. No clinical thresholds and no model logic appear in this file.
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, redirect, render_template, request, url_for

from app.engine.interview import QUESTIONS, InterviewError, accuracy_note, derive_parameters
from app.engine.narrative import compose_rule_based_narrative, get_narrative_engine
from app.engine.predictor import ModelNotTrainedError, get_predictor
from app.engine.validation import validate_submission
from app.knowledgebase.clinical_reference import FEATURE_ORDER, get_reference

logger = logging.getLogger(__name__)

pages_bp = Blueprint("pages", __name__)


#: Icon assigned to each parameter, shared by the form and the results table.
PARAMETER_ICONS = {
    "Pregnancies": "users",
    "Glucose": "droplet",
    "BloodPressure": "activity",
    "SkinThickness": "ruler",
    "Insulin": "zap",
    "BMI": "scale",
    "DiabetesPedigreeFunction": "dna",
    "Age": "calendar",
}

#: The form is split into two fieldsets so eight inputs do not read as one wall.
FIELD_GROUPS = (
    ("Laboratory measurements", "droplet", ("Glucose", "Insulin", "BloodPressure", "SkinThickness")),
    ("Body and history", "users", ("BMI", "Age", "Pregnancies", "DiabetesPedigreeFunction")),
)


def _form_fields():
    """Field descriptors for the assessment form, built from the knowledge base."""
    reference = get_reference()
    fields = []
    for key in FEATURE_ORDER:
        spec = reference.parameter(key)
        low, high = reference.input_range(key)
        fields.append(
            {
                "name": key,
                "label": spec["display_name"],
                "clinical_name": spec["clinical_name"],
                "unit": "" if spec["unit"] == "count" else spec["unit"],
                "hint": spec["measurement_note"],
                "min": low,
                "max": high,
                # Whole numbers for counts and ages, finer steps for lab values.
                "step": "1" if key in {"Pregnancies", "Age"} else (
                    "0.01" if key == "DiabetesPedigreeFunction" else "0.1"
                ),
                "is_diagnostic": spec.get("is_diagnostic", False),
                "icon": PARAMETER_ICONS.get(key, "info"),
            }
        )
    return fields


def _grouped_fields():
    """The same descriptors, arranged into the two form sections."""
    by_name = {field["name"]: field for field in _form_fields()}
    return [
        {
            "title": title,
            "icon": group_icon,
            "fields": [by_name[name] for name in names],
        }
        for title, group_icon, names in FIELD_GROUPS
    ]


#: Pre-filled parameter sets offered on the form, so the tool can be
#: demonstrated without inventing plausible clinical values by hand.
SAMPLE_PROFILES = {
    "healthy": {
        "label": "Low-risk profile",
        "values": {
            "Pregnancies": 1, "Glucose": 92, "BloodPressure": 68, "SkinThickness": 20,
            "Insulin": 80, "BMI": 22.1, "DiabetesPedigreeFunction": 0.18, "Age": 24,
        },
    },
    "borderline": {
        "label": "Borderline profile",
        "values": {
            "Pregnancies": 3, "Glucose": 148, "BloodPressure": 84, "SkinThickness": 33,
            "Insulin": 150, "BMI": 29.4, "DiabetesPedigreeFunction": 0.52, "Age": 41,
        },
    },
    "elevated": {
        "label": "High-risk profile",
        "values": {
            "Pregnancies": 8, "Glucose": 189, "BloodPressure": 96, "SkinThickness": 45,
            "Insulin": 320, "BMI": 38.9, "DiabetesPedigreeFunction": 1.31, "Age": 54,
        },
    },
}


@pages_bp.route("/health")
def health():
    """
    Live AI status check.

        {"status":"operational","llm":"connected",
         "model":"openai/gpt-oss-120b","provider":"groq"}

    Pings the LLM provider and reports whether it is genuinely usable, so an
    expired key, an exhausted quota, or a model the provider has retired shows
    up here instead of silently degrading every screening to the offline
    writer.

    `llm` is one of:
        connected        working normally
        not_configured   no API key set
        disabled         switched off via LLM_ENABLED / LLM_PROVIDER=none
        unauthorized     key rejected - wrong, revoked, or expired
        rate_limited     quota or rate limit reached
        model_not_found  the configured model is no longer offered
        unreachable      network or DNS failure
        provider_error   the provider returned a 5xx
        error            anything else, see `detail`

    Returns 200 when connected and 503 otherwise, so uptime monitors can watch
    it directly. Add `?force=1` to bypass the 30-second cache.
    """
    status = get_narrative_engine().check_provider(
        force=request.args.get("force") in {"1", "true", "yes"}
    )
    return jsonify(status.as_dict()), (200 if status.is_connected else 503)


@pages_bp.route("/")
def index():
    """Landing page: what the tool is, how it works, and what it is not."""
    predictor = get_predictor()
    evaluation = predictor.evaluation.get("screening_metrics") or predictor.evaluation.get(
        "test_metrics", {}
    )
    return render_template(
        "pages/index.html",
        page_title="Early Diabetes Detection",
        metrics=evaluation,
        cross_validation=predictor.evaluation.get("cross_validation", {}),
        dataset=predictor.metadata.get("dataset", {}),
        importances=predictor.feature_importances[:4],
        roc_curve=predictor.evaluation.get("roc_curve", []),
        roc_auc=predictor.evaluation.get("test_metrics", {}).get("roc_auc", 0),
        model_ready=predictor.is_ready,
    )


@pages_bp.route("/check", methods=["GET", "POST"])
def check():
    """
    The plain-language interview, for people without lab results.

    Asks only what someone can actually answer, derives the eight model inputs
    from those answers, and scores them with the same model the clinical
    assessment uses. Values nobody can know are marked unmeasured and imputed.
    """
    if request.method == "GET":
        return render_template(
            "pages/check.html",
            page_title="Check your risk",
            questions=QUESTIONS,
            answers={},
            error=None,
            model_ready=get_predictor().is_ready,
        )

    answers = request.form.to_dict()
    try:
        values, unknown = derive_parameters(answers)
    except InterviewError as exc:
        return (
            render_template(
                "pages/check.html",
                page_title="Check your risk",
                questions=QUESTIONS,
                answers=answers,
                error=str(exc),
                model_ready=get_predictor().is_ready,
            ),
            400,
        )

    predictor = get_predictor()
    try:
        prediction = predictor.predict(values, unknown=unknown)
    except ModelNotTrainedError:
        logger.exception("Interview scored with no trained model on disk.")
        return render_template("errors/503.html", page_title="Model Unavailable"), 503

    return _render_result(
        prediction, predictor, estimate_note=accuracy_note(unknown), source="check"
    )


@pages_bp.route("/assess")
def assess():
    """The 8-parameter assessment form."""
    return render_template(
        "pages/assess.html",
        page_title="Risk Assessment",
        field_groups=_grouped_fields(),
        values={},
        errors={},
        samples=SAMPLE_PROFILES,
        model_ready=get_predictor().is_ready,
    )


@pages_bp.route("/result", methods=["GET", "POST"])
def result():
    """
    Score a submission and render the results dashboard.

    A GET has nothing to score, so it returns to the form. On validation
    failure the form is re-rendered with per-field messages and the values the
    visitor already typed, so nothing has to be entered twice.
    """
    if request.method == "GET":
        return redirect(url_for("pages.assess"))

    submitted = request.form.to_dict()
    cleaned, errors = validate_submission(submitted)

    if errors:
        return (
            render_template(
                "pages/assess.html",
                page_title="Risk Assessment",
                field_groups=_grouped_fields(),
                values=submitted,
                errors=errors,
                samples=SAMPLE_PROFILES,
                model_ready=get_predictor().is_ready,
            ),
            400,
        )

    predictor = get_predictor()
    try:
        prediction = predictor.predict(cleaned)
    except ModelNotTrainedError:
        logger.exception("Prediction attempted with no trained model on disk.")
        return render_template("errors/503.html", page_title="Model Unavailable"), 503

    return _render_result(prediction, predictor, source="assess")


def _render_result(prediction, predictor, estimate_note=None, source="assess"):
    """Render the results dashboard. Shared by the clinical form and the interview."""
    fallback = compose_rule_based_narrative(
        prediction.probability, prediction.findings, prediction.top_drivers
    )

    # Global feature importances, so the findings table can show how much the
    # model relies on each parameter next to the visitor's own value.
    importance_map = {
        row["feature"]: row["importance"] for row in predictor.feature_importances
    }
    peak_importance = max(importance_map.values(), default=1.0) or 1.0

    return render_template(
        "pages/result.html",
        page_title="Your result",
        prediction=prediction,
        fallback_narrative=fallback,
        risk_bands=get_reference().risk_bands,
        guidance=get_reference().general_guidance,
        parameter_icons=PARAMETER_ICONS,
        importance_map=importance_map,
        peak_importance=peak_importance,
        estimate_note=estimate_note,
        source=source,
        parameter_names={
            key: get_reference().parameter(key)["display_name"] for key in FEATURE_ORDER
        },
    )


@pages_bp.route("/clinical-guidelines")
def guidelines():
    """Human-readable view of the WHO/IDF knowledge base."""
    reference = get_reference()
    return render_template(
        "pages/guidelines.html",
        page_title="Clinical Reference",
        reference=reference,
        parameters=[(key, reference.parameter(key)) for key in FEATURE_ORDER],
    )


@pages_bp.route("/model-card")
def model_card():
    """Transparency page: how the model was trained and how well it performs."""
    predictor = get_predictor()
    return render_template(
        "pages/model_card.html",
        page_title="Model Card",
        metadata=predictor.metadata,
        evaluation=predictor.evaluation,
        importances=predictor.feature_importances,
        model_ready=predictor.is_ready,
    )


@pages_bp.route("/about")
def about():
    """Project context, architecture, and limitations."""
    return render_template(
        "pages/about.html",
        page_title="About This Project",
        regional=get_reference().regional_context,
        metadata=get_predictor().metadata,
    )
