"""
JSON API.

Two of these endpoints back the front-end (`/api/narrative` is called by
result.js once the result page has painted). All three are documented on the
Model Card page and are usable by any client.
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from app.config.settings import get_config
from app.engine.chat import ChatUnavailable, get_chat_engine
from app.engine.narrative import get_narrative_engine
from app.engine.predictor import ModelNotTrainedError, get_predictor
from app.engine.validation import ValidationError, validate_or_raise
from app.knowledgebase.clinical_reference import FEATURE_ORDER, get_reference

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")


def _payload():
    """Accept either a JSON body or a form-encoded body."""
    if request.is_json:
        data = request.get_json(silent=True)
        return data if isinstance(data, dict) else {}
    return request.form.to_dict()


@api_bp.route("/health")
def health():
    """
    Full system readiness: the ML model, the LLM, and the knowledge base.

    The LLM section is a live check against the provider (cached briefly), so
    this reports what is actually working, not merely what is configured.
    """
    predictor = get_predictor()
    engine = get_narrative_engine()
    config = get_config()

    provider_status = engine.check_provider()
    model_ready = predictor.is_ready

    # The app is usable whenever the ML model is loaded: a dead LLM only costs
    # the AI-written wording, not the screening itself.
    if model_ready and provider_status.is_connected:
        overall = "operational"
    elif model_ready:
        overall = "degraded"
    else:
        overall = "down"

    return (
        jsonify(
            {
                "status": overall,
                "model": {
                    "loaded": model_ready,
                    "version": predictor.metadata.get("model_version"),
                    "algorithm": predictor.metadata.get("algorithm"),
                    "trained_at": predictor.metadata.get("trained_at"),
                    "screening_threshold": predictor.screening_threshold,
                },
                "narrative": {
                    "llm": provider_status.llm,
                    "provider": provider_status.provider,
                    "model": provider_status.model,
                    "detail": provider_status.detail,
                    "latency_ms": provider_status.latency_ms,
                    "checked_at": provider_status.checked_at,
                    "fallback": "knowledge-base writer (always available)",
                },
                "knowledge_base": {
                    "parameters": len(FEATURE_ORDER),
                    "last_reviewed": get_reference().raw.get("last_reviewed"),
                },
                "environment": config.env,
            }
        ),
        200 if model_ready else 503,
    )


@api_bp.route("/predict", methods=["POST"])
def predict():
    """
    Score an 8-parameter submission.

    Returns the probability, the risk band, and the WHO/IDF reading of each
    value. Deterministic - no LLM call is made here.
    """
    try:
        cleaned = validate_or_raise(_payload())
    except ValidationError as exc:
        return jsonify({"error": "validation_failed", "fields": exc.errors}), 400

    try:
        prediction = get_predictor().predict(cleaned)
    except ModelNotTrainedError as exc:
        logger.exception("Prediction requested with no trained model.")
        return jsonify({"error": "model_unavailable", "message": str(exc)}), 503

    return jsonify(prediction.as_dict())


@api_bp.route("/narrative", methods=["POST"])
def narrative():
    """
    Score a submission and return the natural-language explanation.

    Called by the result page after first paint. Never fails on a provider
    error: the knowledge-base narrative is returned instead, flagged with
    `is_ai_generated: false` so the interface can disclose it.
    """
    try:
        cleaned = validate_or_raise(_payload())
    except ValidationError as exc:
        return jsonify({"error": "validation_failed", "fields": exc.errors}), 400

    try:
        prediction = get_predictor().predict(cleaned)
    except ModelNotTrainedError as exc:
        logger.exception("Narrative requested with no trained model.")
        return jsonify({"error": "model_unavailable", "message": str(exc)}), 503

    result = get_narrative_engine().generate(
        prediction.probability, prediction.findings, prediction.top_drivers
    )

    return jsonify(
        {
            "probability": round(prediction.probability, 4),
            "percentage": prediction.percentage,
            "risk_band": {"id": prediction.band_id, "label": prediction.band_label},
            "narrative": result.as_dict(),
        }
    )


@api_bp.route("/chat", methods=["POST"])
def chat():
    """
    One turn of conversation with the assistant.

    Stateless: the browser sends the whole history each time, so nothing about a
    conversation is retained on the server - the same promise the rest of the
    app makes about health data.

    The assistant cannot produce a risk figure on its own. When it decides it
    has enough to score someone it calls the Random Forest, and the resulting
    prediction is returned alongside the reply so the interface can show the
    number from the model rather than from the model's prose.
    """
    payload = _payload()
    history = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(history, list):
        return jsonify({"error": "bad_request", "message": "Expected a messages array."}), 400

    try:
        result = get_chat_engine().reply(history)
    except ChatUnavailable as exc:
        # A friendly, user-facing sentence rather than an error state.
        return jsonify({"reply": str(exc), "prediction": None, "degraded": True}), 200

    return jsonify({**result, "degraded": False})
