"""Configuration layer: environment loading, app settings, and security headers."""

from app.config.settings import Config, get_config

__all__ = ["Config", "get_config"]
