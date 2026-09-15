"""
The conversational front door.

A third way into the same model. The clinical form takes eight numbers, the
interview asks ten plain questions, and this lets someone simply describe
themselves: "I'm 47, about 102kg and 1.7m, my mum and brother are diabetic."

THE RULE THAT MATTERS
---------------------
The language model NEVER produces the risk figure. It collects values from the
conversation and calls `predict_diabetes_risk`, which runs the same Random
Forest the other two routes use; only then does it explain what came back. A
model that estimated risk itself would sound completely convincing and be
untethered from anything measured, which is the exact failure this project is
built to avoid.

So the split is the same as everywhere else in the app: the forest does the
arithmetic, the knowledge base supplies the clinical facts, and the language
model is a writer and a parser - nothing more.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.config.settings import Config, get_config
from app.engine.interview import PEDIGREE_BY_FAMILY_HISTORY, InterviewError, derive_parameters
from app.engine.narrative import LLMError, USER_AGENT, _post_json, PROVIDER_ENDPOINTS
from app.engine.predictor import ModelNotTrainedError, get_predictor
from app.knowledgebase.clinical_reference import get_reference

logger = logging.getLogger(__name__)

#: Longest single message accepted from the browser.
MAX_MESSAGE_CHARS = 800

#: How many past turns are replayed. Chat resends its history on every request,
#: so this is what keeps a long conversation inside the provider's per-minute
#: token allowance.
MAX_HISTORY_TURNS = 12

#: Tool round-trips permitted per request. One is enough to score a person;
#: the second exists only so a malformed first call can be retried.
MAX_TOOL_ROUNDS = 2

MAX_OUTPUT_TOKENS = 900


SYSTEM_PROMPT = """You are the assistant inside a diabetes screening tool used in Nigeria \
by health workers and by members of the public.

THE ONE UNBREAKABLE RULE
You do not estimate diabetes risk yourself. Ever. You have a tool called \
`predict_diabetes_risk` that runs a trained Random Forest model. Any figure you give must \
have come from that tool. If you have not called it, you have no number, and you say so.

WHAT YOU DO
1. If someone wants their risk checked, gather what the tool needs through normal \
conversation: their age, their height, their weight, whether they have ever been pregnant \
(and how many times), and whether diabetes runs in their close family. Ask for a few at a \
time, not one by one, and never re-ask something they already told you.
2. Blood pressure and a blood sugar reading are optional. Ask once whether they know \
them. If they don't, say that is fine and carry on - the check still works.
3. Once you have age, height, weight and family history, call the tool.
4. When the tool returns, explain the result in plain words: the percentage, what band it \
falls in, and the recommended next step it gave you.
5. You can also answer general questions about diabetes, about the eight measurements, \
and about how this tool works.

HOW YOU TALK
- Plain English for someone with no medical training. Short sentences.
- Warm and calm. A high result should read as urgent and actionable, never frightening.
- Keep replies short - two or three sentences unless explaining a result.
- No markdown, no bullet symbols, no headings. Just sentences.
- British spelling.

HARD LIMITS
- Never diagnose. This tool screens. Say "suggests", "is associated with", "worth getting \
tested for" - never "you have diabetes".
- Never name a medicine, a dose, or a treatment. You may suggest tests, lifestyle change, \
and seeing a clinician.
- Never invent a clinical threshold or statistic. If you were not given a fact, say you \
don't have it and point to the Clinical reference page.
- If someone describes an emergency - chest pain, collapse, confusion, vomiting that will \
not stop - tell them to seek urgent care now, and stop screening them.
- If asked something unrelated to diabetes or this tool, say briefly that this is what you \
can help with, and offer to check their risk."""


#: The one tool. Parameters are the plain-language ones a person can answer,
#: not the eight clinical inputs - the derivation lives in `interview.py` and is
#: shared with the step-by-step page, so both routes score identically.
PREDICT_TOOL = {
    "type": "function",
    "function": {
        "name": "predict_diabetes_risk",
        "description": (
            "Score a person's diabetes risk using the trained Random Forest model. "
            "Call this once you know at least age, height, weight and family history. "
            "Returns the probability, the risk band, and the recommended next step."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "age": {"type": "number", "description": "Age in years, 18-100."},
                "height_cm": {"type": "number", "description": "Height in centimetres."},
                "weight_kg": {"type": "number", "description": "Weight in kilograms."},
                "family_history": {
                    "type": "string",
                    "enum": list(PEDIGREE_BY_FAMILY_HISTORY),
                    "description": (
                        "Diabetes among close relatives: 'none', 'one' for a single parent "
                        "or sibling, 'several' for several close relatives, "
                        "'several_young' if several were also diagnosed young."
                    ),
                },
                "been_pregnant": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Whether the person has ever been pregnant. Assume 'no' for men.",
                },
                "pregnancies": {
                    "type": "number",
                    "description": "Number of pregnancies. Only when been_pregnant is 'yes'.",
                },
                "blood_pressure": {
                    "type": "number",
                    "description": (
                        "Optional. The lower (diastolic) number of a blood pressure reading, "
                        "in mm Hg. Omit if unknown."
                    ),
                },
                "glucose": {
                    "type": "number",
                    "description": (
                        "Optional but valuable. A blood sugar reading in mg/dL. "
                        "Omit if the person has never had one."
                    ),
                },
            },
            "required": ["age", "height_cm", "weight_kg", "family_history"],
        },
    },
}


class ChatUnavailable(RuntimeError):
    """Raised when the conversation cannot proceed. Always shown to the user."""


# --------------------------------------------------------------------------
# The tool
# --------------------------------------------------------------------------

def run_prediction(arguments: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Execute the tool call: derive the eight inputs, then score them.

    Returns `(tool_result, prediction_payload)`. The first goes back to the
    language model as facts to explain; the second goes to the browser so the
    widget can show the same figure without trusting the model to repeat it
    accurately.
    """
    answers = {
        "age": arguments.get("age"),
        "height_cm": arguments.get("height_cm"),
        "weight_kg": arguments.get("weight_kg"),
        "family_history": arguments.get("family_history"),
        "been_pregnant": str(arguments.get("been_pregnant", "no")).lower(),
        "pregnancies": arguments.get("pregnancies"),
        # The interview asks these as yes/no gates; here their presence is the gate.
        "knows_bp": "yes" if arguments.get("blood_pressure") is not None else "no",
        "blood_pressure": arguments.get("blood_pressure"),
        "had_glucose_test": "yes" if arguments.get("glucose") is not None else "no",
        "glucose": arguments.get("glucose"),
    }

    try:
        values, unknown = derive_parameters(answers)
    except InterviewError as exc:
        # Handed back to the model so it can ask the person to clarify.
        return {"error": str(exc)}, None

    try:
        prediction = get_predictor().predict(values, unknown=unknown)
    except ModelNotTrainedError:
        return {"error": "The risk model is not available right now."}, None

    reference = get_reference()
    flagged = [
        {
            "measurement": finding.display_name,
            "value": f"{finding.value:g} {finding.unit}".strip(),
            "reading": finding.band_label,
        }
        for finding in prediction.findings
        if finding.is_flagged
    ][:3]

    tool_result = {
        "probability_percent": prediction.percentage,
        "risk_band": prediction.band_label,
        "what_the_band_means": prediction.band_summary,
        "recommended_next_step": prediction.band_action,
        "measurements_outside_normal_range": flagged,
        "estimated_not_measured": [
            reference.parameter(key)["display_name"] for key in prediction.unknown
        ],
        "reminder": (
            "This is a screening estimate, not a diagnosis. Only a laboratory test can "
            "diagnose diabetes."
        ),
    }
    return tool_result, prediction.as_dict()


# --------------------------------------------------------------------------
# The conversation
# --------------------------------------------------------------------------

def _sanitise(history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Keep only well-formed recent turns, and bound their size."""
    clean: List[Dict[str, str]] = []
    for entry in history[-MAX_HISTORY_TURNS:]:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        content = entry.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        text = content.strip()[:MAX_MESSAGE_CHARS]
        if text:
            clean.append({"role": role, "content": text})
    return clean


class ChatEngine:
    """Runs one turn of conversation, including any tool call it triggers."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()

    @property
    def is_available(self) -> bool:
        # Tool calling is an OpenAI-compatible feature; Gemini's shape differs,
        # so the chatbot is offered only on the providers that share it.
        return self.config.llm_configured and self.config.llm_provider in ("groq", "openai")

    def _call(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.config.llm_model or "openai/gpt-oss-120b",
            "messages": messages,
            "tools": [PREDICT_TOOL],
            "tool_choice": "auto",
            "temperature": 0.3,
            "max_completion_tokens": MAX_OUTPUT_TOKENS,
        }
        data = _post_json(
            PROVIDER_ENDPOINTS[self.config.llm_provider],
            payload,
            {"Authorization": f"Bearer {self.config.llm_api_key}",
             "User-Agent": USER_AGENT},
            self.config.llm_timeout,
        )
        try:
            return data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected chat response: {str(data)[:200]}") from exc

    def reply(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Produce the assistant's next message.

        Returns `{"reply": str, "prediction": dict | None}`. Raises
        `ChatUnavailable` with a message safe to show the user.
        """
        if not self.is_available:
            raise ChatUnavailable(
                "The chat assistant needs an API key. You can still check your risk with "
                "the step-by-step questions."
            )

        turns = _sanitise(history)
        if not turns:
            raise ChatUnavailable("Please type a message.")

        messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}] + turns
        prediction: Optional[Dict[str, Any]] = None

        for _ in range(MAX_TOOL_ROUNDS):
            try:
                message = self._call(messages)
            except LLMError as exc:
                text = str(exc)
                if "429" in text or "rate_limit" in text.lower():
                    raise ChatUnavailable(
                        "The free AI allowance is used up for the moment. Wait a minute and "
                        "try again, or use the step-by-step check, which never needs the AI."
                    ) from exc
                logger.warning("Chat provider failed: %s", exc)
                raise ChatUnavailable(
                    "The assistant is unavailable right now. The step-by-step check still "
                    "works and does not need it."
                ) from exc

            calls = message.get("tool_calls")
            if not calls:
                return {"reply": (message.get("content") or "").strip(), "prediction": prediction}

            messages.append(message)
            for call in calls:
                function = call.get("function", {})
                if function.get("name") != "predict_diabetes_risk":
                    result: Dict[str, Any] = {"error": "Unknown tool."}
                else:
                    try:
                        arguments = json.loads(function.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                    result, payload = run_prediction(arguments)
                    if payload:
                        prediction = payload

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": json.dumps(result),
                })

        # Two rounds without a plain answer: give the person something useful.
        return {
            "reply": (
                "I could not complete that. You can check your risk with the step-by-step "
                "questions instead."
            ),
            "prediction": prediction,
        }


_engine: Optional[ChatEngine] = None


def get_chat_engine() -> ChatEngine:
    global _engine
    if _engine is None:
        _engine = ChatEngine()
    return _engine
