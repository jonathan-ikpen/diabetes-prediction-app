"""
The language half of the hybrid architecture.

Takes the deterministic output of `engine.predictor` and turns it into a
clinical narrative a patient can read, grounded in the WHO/IDF facts assembled
by `knowledgebase.prompt_builder`.

Provider: Groq, using its free hosted open models (default
`openai/gpt-oss-120b`). Groq exposes an OpenAI-compatible chat-completions
endpoint, so the same request shape also works against OpenAI itself, and a
Google Gemini client is included as a third option.

Critically, there is also a deterministic rule-based writer that composes the
same three paragraphs directly from the knowledge base. It runs whenever no API
key is configured, the provider errors, or the request times out, so the
application NEVER fails to explain a result and can be demonstrated completely
offline.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.config.settings import Config, get_config
from app.knowledgebase.clinical_reference import (
    ParameterFinding,
    get_reference,
    rank_by_influence,
)
from app.knowledgebase.prompt_builder import SYSTEM_PROMPT, build_narrative_prompt

logger = logging.getLogger(__name__)

#: Chat-completions endpoint per provider. Groq and OpenAI share a request
#: shape; Gemini has its own.
PROVIDER_ENDPOINTS = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openai": "https://api.openai.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
}

#: Model-listing endpoint, used by the health check to confirm in one cheap call
#: that the key is valid AND the configured model still exists.
PROVIDER_MODEL_LISTS = {
    "groq": "https://api.groq.com/openai/v1/models",
    "openai": "https://api.openai.com/v1/models",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
}

#: How long a health result is reused before the provider is asked again.
#: Stops a refreshed dashboard from hammering the API.
HEALTH_CACHE_SECONDS = 30

#: Sent on every provider request.
#:
#: This is not cosmetic. Groq sits behind Cloudflare, which rejects Python's
#: default `Python-urllib/3.x` agent outright with `403 error code: 1010` -
#: a block that looks exactly like a bad API key but is not one. Identifying
#: the client properly is what makes the request go through.
USER_AGENT = "diabetes-risk-app/1.0 (+screening tool; python-urllib)"

#: Used when the operator does not pin one in LLM_MODEL.
DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
}

#: Generous enough for three paragraphs plus any reasoning overhead, tight
#: enough that a runaway generation cannot stall a web request.
MAX_OUTPUT_TOKENS = 4000

#: Slightly conservative: this is a rewriting task, not a creative one.
TEMPERATURE = 0.4


@dataclass
class NarrativeResult:
    """A generated explanation plus the provenance the UI has to disclose."""

    text: str
    source: str            # "groq" | "openai" | "gemini" | "rule-based"
    model: str             # model id, or "knowledge-base-template"
    is_ai_generated: bool
    fallback_reason: Optional[str] = None

    @property
    def paragraphs(self) -> List[str]:
        """Split into paragraphs for templating, blank entries removed."""
        return [block.strip() for block in re.split(r"\n\s*\n", self.text) if block.strip()]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "paragraphs": self.paragraphs,
            "source": self.source,
            "model": self.model,
            "is_ai_generated": self.is_ai_generated,
            "fallback_reason": self.fallback_reason,
        }


class LLMError(RuntimeError):
    """Any provider-side failure. Always caught - it never reaches the user."""


@dataclass
class ProviderStatus:
    """The live result of pinging the LLM provider."""

    llm: str               # "connected" | "not_configured" | "disabled" | "unauthorized"
                           # | "model_not_found" | "rate_limited" | "provider_error"
                           # | "unreachable" | "error"
    provider: str
    model: str
    detail: Optional[str] = None
    latency_ms: Optional[int] = None
    checked_at: Optional[str] = None

    @property
    def is_connected(self) -> bool:
        return self.llm == "connected"

    def as_dict(self) -> Dict[str, Any]:
        """The compact shape served at /health."""
        payload: Dict[str, Any] = {
            "status": "operational" if self.is_connected else "degraded",
            "llm": self.llm,
            "model": self.model,
            "provider": self.provider,
        }
        if self.detail:
            payload["detail"] = self.detail
        if self.latency_ms is not None:
            payload["latency_ms"] = self.latency_ms
        if not self.is_connected:
            # The app never stops working when the LLM does; say so explicitly.
            payload["fallback"] = "knowledge-base writer active"
        return payload


# --------------------------------------------------------------------------
# Deterministic writer - the safety net
# --------------------------------------------------------------------------

#: Units that read as noise inside a sentence ("a parity of 6 count").
_UNITLESS = frozenset({"count", "score"})


def _measurement_phrase(finding: ParameterFinding) -> str:
    """e.g. 'a plasma glucose of 168 mg/dL (Impaired Glucose Tolerance)'."""
    unit = "" if finding.unit in _UNITLESS else f" {finding.unit}"
    # Band labels carry roman numerals ("Obesity Class II"), so they are left
    # in their authored casing rather than lowercased.
    return f"{finding.display_name.lower()} of {finding.value:g}{unit} ({finding.band_label})"


def _describe_findings(findings: List[ParameterFinding], limit: int = 3) -> str:
    """Readable sentence listing the most influential abnormal measurements."""
    flagged = [f for f in findings if f.is_flagged][:limit]
    if not flagged:
        return "None of the eight measurements fell outside their WHO or IDF reference range."

    clauses = [_measurement_phrase(finding) for finding in flagged]
    listed = clauses[0] if len(clauses) == 1 else ", ".join(clauses[:-1]) + f", and {clauses[-1]}"
    verb = "was" if len(clauses) == 1 else "were"
    return f"The measurements that most influenced this result {verb} a {listed}."


def compose_rule_based_narrative(
    probability: float,
    findings: List[ParameterFinding],
    top_drivers: Optional[List[Dict[str, Any]]] = None,
) -> NarrativeResult:
    """
    Build the three-paragraph explanation directly from the knowledge base.

    Produces the same structure the LLM is asked for, using only facts already
    present in `who_idf_thresholds.json`. Fully deterministic and offline.
    """
    reference = get_reference()
    band = reference.band_for_probability(probability)
    percent = round(probability * 100, 1)
    ranked = rank_by_influence(findings, top_drivers)
    flagged = [f for f in ranked if f.is_flagged]

    first = (
        f"Based on the eight measurements provided, this screening places the result in the "
        f"{band['label'].lower()} category, with a calculated probability of {percent}%. "
        f"{band['summary']} This figure describes how closely the profile resembles people "
        f"who were later found to have diabetes in the reference population; it is not a "
        f"diagnosis and it is not a prediction of what will happen."
    )

    if flagged:
        leading = flagged[0]
        second_parts = [
            _describe_findings(ranked),
            f"For {leading.display_name.lower()}, the {leading.band_label} band is "
            f"defined as: {leading.note}",
            "Every value was compared against the World Health Organization and "
            "International Diabetes Federation thresholds held in this tool's clinical "
            "reference.",
        ]
    else:
        second_parts = [
            "All eight measurements sat within their World Health Organization and "
            "International Diabetes Federation reference ranges, which is the main "
            "reason the calculated risk is low.",
            "No single parameter stood out as a concern on this screening.",
        ]
    second = " ".join(second_parts)

    lifestyle = next(
        (item for item in reference.general_guidance if item["id"] == "lifestyle"), None
    )
    third = (
        f"{band['action']} "
        f"{lifestyle['body'] if lifestyle else ''} "
        f"This tool screens rather than diagnoses, so please discuss this result with a "
        f"qualified healthcare professional before drawing any conclusion from it."
    ).strip()

    return NarrativeResult(
        text=f"{first}\n\n{second}\n\n{third}",
        source="rule-based",
        model="knowledge-base-template",
        is_ai_generated=False,
    )


# --------------------------------------------------------------------------
# Provider clients
# --------------------------------------------------------------------------

def _post_json(
    url: str, payload: Dict[str, Any], headers: Dict[str, str], timeout: int
) -> Dict[str, Any]:
    """Minimal JSON POST. All three providers speak plain HTTPS + JSON."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            **headers,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise LLMError(f"HTTP {exc.code} from provider: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LLMError(f"Provider request failed: {exc}") from exc


def _get_json(url: str, headers: Dict[str, str], timeout: int) -> Any:
    """GET returning parsed JSON. Used by the health check only."""
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, **headers}, method="GET"
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class NarrativeEngine:
    """Routes a narrative request to the configured provider, with fallback."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or get_config()
        self._status_cache: Optional[tuple[float, ProviderStatus]] = None
        self._status_lock = threading.Lock()

    @property
    def provider(self) -> str:
        return self.config.llm_provider

    @property
    def model_id(self) -> str:
        return self.config.llm_model or DEFAULT_MODELS.get(self.provider, "")

    @property
    def is_available(self) -> bool:
        return self.config.llm_configured and self.provider in DEFAULT_MODELS

    # ------------------------------------------------------------ health check

    def check_provider(self, force: bool = False) -> ProviderStatus:
        """
        Ask the provider whether it is actually usable right now.

        Lists the provider's models, which in one cheap call proves three things
        at once: the network is reachable, the API key is accepted, and the
        configured model still exists. That covers the failures that matter -
        a revoked key, an exhausted quota, and a model Groq has renamed or
        retired out from under us.

        Never raises. Results are cached for HEALTH_CACHE_SECONDS so a refreshed
        status page does not hammer the API.
        """
        if not force:
            with self._status_lock:
                cached = self._status_cache
            if cached and (time.monotonic() - cached[0]) < HEALTH_CACHE_SECONDS:
                return cached[1]

        status = self._probe_provider()
        with self._status_lock:
            self._status_cache = (time.monotonic(), status)
        return status

    def _probe_provider(self) -> ProviderStatus:
        provider, model = self.provider, self.model_id
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        def result(llm: str, detail: Optional[str] = None, latency: Optional[int] = None):
            return ProviderStatus(
                llm=llm, provider=provider, model=model,
                detail=detail, latency_ms=latency, checked_at=stamp,
            )

        if not self.config.llm_enabled:
            return result("disabled", "LLM_ENABLED is false.")
        if provider in {"none", ""}:
            return result("disabled", "LLM_PROVIDER is 'none'.")
        if provider not in PROVIDER_MODEL_LISTS:
            return result("error", f"Unknown provider '{provider}'.")
        if not self.config.llm_api_key:
            return result("not_configured", "No API key set. Add LLM_API_KEY to your .env file.")

        url = PROVIDER_MODEL_LISTS[provider]
        headers = (
            {"x-goog-api-key": self.config.llm_api_key}
            if provider == "gemini"
            else {"Authorization": f"Bearer {self.config.llm_api_key}"}
        )

        started = time.monotonic()
        try:
            data = _get_json(url, headers, min(self.config.llm_timeout, 15))
        except urllib.error.HTTPError as exc:
            latency = int((time.monotonic() - started) * 1000)
            body = exc.read().decode("utf-8", errors="replace")[:200]
            if exc.code in (401, 403):
                return result("unauthorized", f"API key rejected ({exc.code}). {body}", latency)
            if exc.code == 429:
                return result("rate_limited", f"Quota or rate limit reached. {body}", latency)
            if exc.code >= 500:
                return result("provider_error", f"{provider} returned {exc.code}.", latency)
            return result("error", f"HTTP {exc.code}. {body}", latency)
        except (urllib.error.URLError, TimeoutError) as exc:
            latency = int((time.monotonic() - started) * 1000)
            return result("unreachable", f"Could not reach {provider}: {exc}", latency)
        except (json.JSONDecodeError, ValueError) as exc:
            return result("error", f"Unreadable response from {provider}: {exc}")

        latency = int((time.monotonic() - started) * 1000)

        # Both response shapes carry a list of objects with an id/name field.
        entries = data.get("data") or data.get("models") or []
        available = {
            str(item.get("id") or item.get("name", "")).removeprefix("models/")
            for item in entries
            if isinstance(item, dict)
        }

        if available and model not in available:
            return result(
                "model_not_found",
                f"'{model}' is not offered by {provider} any more. "
                f"{len(available)} other models are available - set LLM_MODEL to one of them.",
                latency,
            )

        return result("connected", None, latency)

    # ------------------------------------------------- Groq / OpenAI (shared)

    def _call_openai_compatible(self, prompt: str) -> str:
        """
        Groq and OpenAI expose the same chat-completions contract.

        Groq additionally accepts `reasoning_effort` on the gpt-oss models,
        which is sent only for Groq so an OpenAI request stays valid.
        """
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": TEMPERATURE,
            "max_completion_tokens": MAX_OUTPUT_TOKENS,
        }
        if self.provider == "groq":
            payload["reasoning_effort"] = self.config.llm_effort

        data = _post_json(
            PROVIDER_ENDPOINTS[self.provider],
            payload,
            {"Authorization": f"Bearer {self.config.llm_api_key}"},
            self.config.llm_timeout,
        )

        try:
            choice = data["choices"][0]
            text = (choice["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError(f"Unexpected response shape: {str(data)[:200]}") from exc

        if not text:
            # gpt-oss can spend the whole budget on reasoning and return nothing.
            reason = choice.get("finish_reason", "unknown")
            raise LLMError(f"Provider returned an empty message (finish_reason={reason}).")
        return text

    # ---------------------------------------------------------------- Gemini

    def _call_gemini(self, prompt: str) -> str:
        url = PROVIDER_ENDPOINTS["gemini"].format(model=self.model_id)
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
                "temperature": TEMPERATURE,
            },
        }
        data = _post_json(
            url, payload, {"x-goog-api-key": self.config.llm_api_key}, self.config.llm_timeout
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Unexpected Gemini response shape: {str(data)[:200]}") from exc
        if not text:
            raise LLMError("Gemini returned an empty response.")
        return text

    # ------------------------------------------------------------------ entry

    def generate(
        self,
        probability: float,
        findings: List[ParameterFinding],
        top_drivers: Optional[List[Dict[str, Any]]] = None,
    ) -> NarrativeResult:
        """
        Produce a narrative, degrading to the deterministic writer on any failure.

        This method does not raise. A screening result must always be explained.
        """
        if not self.is_available:
            result = compose_rule_based_narrative(probability, findings, top_drivers)
            result.fallback_reason = (
                "No LLM provider is configured, so the explanation was composed "
                "directly from the WHO/IDF knowledge base."
            )
            return result

        prompt = build_narrative_prompt(probability, findings, top_drivers)
        handlers = {
            "groq": self._call_openai_compatible,
            "openai": self._call_openai_compatible,
            "gemini": self._call_gemini,
        }

        try:
            text = handlers[self.provider](prompt)
            return NarrativeResult(
                text=text,
                source=self.provider,
                model=self.model_id,
                is_ai_generated=True,
            )
        except LLMError as exc:
            logger.warning("LLM narrative generation failed (%s): %s", self.provider, exc)
        except Exception as exc:  # defensive: an unexpected error must not 500
            logger.exception("Unexpected LLM failure (%s): %s", self.provider, exc)

        result = compose_rule_based_narrative(probability, findings, top_drivers)
        result.fallback_reason = (
            "The AI explanation service was unavailable, so this explanation was "
            "composed directly from the WHO/IDF knowledge base instead."
        )
        return result


_engine: Optional[NarrativeEngine] = None
_engine_lock = threading.Lock()


def get_narrative_engine() -> NarrativeEngine:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = NarrativeEngine()
    return _engine


def generate_narrative(
    probability: float,
    findings: List[ParameterFinding],
    top_drivers: Optional[List[Dict[str, Any]]] = None,
) -> NarrativeResult:
    """Convenience wrapper over the shared engine."""
    return get_narrative_engine().generate(probability, findings, top_drivers)
