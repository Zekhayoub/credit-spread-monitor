"""
Centralized configuration loader.

Loads config.yaml and .env from the project root, regardless of the
current working directory. All modules import CONFIG and PROJECT_ROOT
from here.
"""

import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv
import os

logger = logging.getLogger(__name__)

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"


def _load_config() -> dict:
    """Load config.yaml and return as dict."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Configuration loaded from %s", CONFIG_PATH)
    return config


def get_fred_api_key() -> str:
    """Load FRED API key from .env file."""
    load_dotenv(ENV_PATH)
    api_key = os.getenv("FRED_API_KEY")

    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "FRED_API_KEY not set. Copy .env.example to .env and add your key. "
            "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    return api_key

CONFIG = _load_config()