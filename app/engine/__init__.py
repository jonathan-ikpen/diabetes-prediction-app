"""
Inference Engine layer.

Everything that happens between a submitted form and a rendered result:

    validation.py   sanitise and range-check the 8 clinical parameters
    predictor.py    load model.pkl once, produce the mathematical risk output
    narrative.py    turn that output into grounded natural language via an LLM

The engine depends on `knowledgebase` for clinical facts but never on Flask, so
it can be unit-tested and reused outside the web app.
"""

from app.engine.narrative import NarrativeResult, generate_narrative, get_narrative_engine
from app.engine.predictor import PredictionResult, get_predictor
from app.engine.validation import ValidationError, validate_submission

__all__ = [
    "ValidationError",
    "validate_submission",
    "PredictionResult",
    "get_predictor",
    "NarrativeResult",
    "generate_narrative",
    "get_narrative_engine",
]
