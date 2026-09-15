"""
Knowledge Base layer.

Holds every piece of hardcoded clinical knowledge the application relies on:
WHO / IDF thresholds, risk band definitions, and the prompt scaffolding that
grounds the LLM narrative in those facts.

Nothing in this layer imports from `engine` or the Flask app, so the clinical
facts can be reused, tested, or replaced independently of the runtime.
"""

from app.knowledgebase.clinical_reference import (
    ClinicalReference,
    ParameterFinding,
    get_reference,
)
from app.knowledgebase.prompt_builder import build_narrative_prompt, SYSTEM_PROMPT

__all__ = [
    "ClinicalReference",
    "ParameterFinding",
    "get_reference",
    "build_narrative_prompt",
    "SYSTEM_PROMPT",
]
