"""
Application configuration, read from the environment.

Values come from a `.env` file in the project root (see `.env.example`).
No secret is ever hardcoded here or committed; `.env` is git-ignored.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load .env once, at import time. Existing environment variables win, so
# container or CI settings are never clobbered by a stale local file.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


@dataclass
class Config:
    """Runtime settings for the Flask app and the inference engine."""

    # ------------------------------------------------------------------ Flask
    env: str = field(default_factory=lambda: os.getenv("FLASK_ENV", "production"))
    #: A generated key keeps dev working out of the box. Production must set
    #: SECRET_KEY explicitly, or sessions reset on every restart.
    secret_key: str = field(
        default_factory=lambda: os.getenv("SECRET_KEY") or secrets.token_hex(32)
    )
    host: str = field(default_factory=lambda: os.getenv("FLASK_HOST", "127.0.0.1"))
    #: 8137 rather than 5000: Windows reserves parts of the port range and 5000
    #: is frequently unavailable, as are the usual 8000 and 8080.
    port: int = field(default_factory=lambda: _env_int("FLASK_PORT", 8137))

    # -------------------------------------------------------------------- LLM
    #: "groq" | "openai" | "gemini" | "none"
    llm_provider: str = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "groq").strip().lower()
    )
    llm_api_key: str = field(
        default_factory=lambda: (
            os.getenv("LLM_API_KEY", "").strip()
            or os.getenv("GROQ_API_KEY", "").strip()
        )
    )
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "").strip())
    llm_timeout: int = field(default_factory=lambda: _env_int("LLM_TIMEOUT_SECONDS", 45))
    #: Reasoning depth, sent to Groq's gpt-oss models as `reasoning_effort`.
    #: The prompt supplies every clinical fact, so this is a rewriting task
    #: rather than a reasoning one - "low" keeps it inside a web-request
    #: latency budget.
    llm_effort: str = field(
        default_factory=lambda: os.getenv("LLM_EFFORT", "low").strip().lower()
    )
    #: Set false to force the deterministic narrative and skip the network call.
    llm_enabled: bool = field(default_factory=lambda: _env_bool("LLM_ENABLED", True))

    # --------------------------------------------------------------- Security
    #: Largest accepted request body. The form carries 8 short numbers, so this
    #: is far above any legitimate submission and bounds abusive payloads.
    max_content_length: int = 16 * 1024

    @property
    def is_debug(self) -> bool:
        return self.env.lower() in {"development", "dev", "debug"}

    @property
    def llm_configured(self) -> bool:
        """True when a live LLM call is possible and permitted."""
        if not self.llm_enabled or self.llm_provider in {"none", ""}:
            return False
        # Fall back to the provider's conventional env var if LLM_API_KEY is
        # unset, so an existing GROQ_API_KEY in the shell just works.
        return bool(self.llm_api_key or os.getenv(f"{self.llm_provider.upper()}_API_KEY"))

    def security_headers(self) -> Dict[str, str]:
        """
        Baseline response headers applied to every response.

        `script-src 'self'` is strict: the interface ships no third-party and no
        inline JavaScript, which is where XSS actually lives. Fonts are
        self-hosted, so no external origin is needed at all.

        `style-src` allows inline styles for one narrow reason: the result and
        model-card pages set bar widths from server-computed percentages
        (`style="width: 62.4%"`), which cannot be expressed as a fixed class and
        which a nonce cannot cover, since nonces do not apply to style
        attributes. Every other style in the app lives in a stylesheet. The
        residual risk is low - a style-only injection still requires an HTML
        injection first, and all template values are auto-escaped by Jinja.
        """
        csp = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "font-src 'self'; "
            "img-src 'self' data:; "
            "connect-src 'self'; "
            "form-action 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "object-src 'none'"
        )
        headers = {
            "Content-Security-Policy": csp,
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=(), interest-cohort=()",
            # Health data must never sit in a shared cache.
            "Cache-Control": "no-store, max-age=0",
        }
        if not self.is_debug:
            headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return headers


_config: Config | None = None


def get_config() -> Config:
    """Process-wide singleton, so `.env` is parsed once."""
    global _config
    if _config is None:
        _config = Config()
    return _config
