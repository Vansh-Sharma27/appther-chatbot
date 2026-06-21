"""Resolve API keys from AWS Secrets Manager at Lambda cold start.

Environment variables expected:
  SECRET_VOYAGE_ARN  — ARN of the Voyage AI API key secret
  SECRET_GEMINI_ARN  — ARN of the Gemini API key secret
  SECRET_JINA_ARN    — ARN of the Jina AI API key secret (optional)

At module-load time (cold start), all secrets are fetched once and cached in
module globals for the lifetime of the execution environment. On warm starts,
the module globals are reused — no additional Secrets Manager calls.

Usage:
    from api.secrets import get_secrets
    secrets = get_secrets()
    voyage_key = secrets["VOYAGE_API_KEY"]
"""

from __future__ import annotations

import logging
import os

import boto3

logger = logging.getLogger(__name__)

_SECRETS_CACHE: dict[str, str] | None = None


def _resolve_secret(arn: str, client) -> str | None:
    """Fetch a single secret value from Secrets Manager.

    Returns None if the secret ARN is empty, the secret doesn't exist, or
    access is denied. The caller handles missing secrets by falling through
    to env-var based lookups.
    """
    if not arn:
        return None
    try:
        resp = client.get_secret_value(SecretId=arn)
        return resp.get("SecretString")
    except client.exceptions.ResourceNotFoundException:
        logger.warning("Secret not found: %s", arn)
        return None
    except Exception:
        logger.exception("Failed to resolve secret: %s", arn)
        return None


def get_secrets() -> dict[str, str]:
    """Return a dict of resolved secrets.

    Result is cached after the first call. Each key maps to a plaintext value,
    or an empty string if resolution failed but the env var fallback exists.
    """
    global _SECRETS_CACHE
    if _SECRETS_CACHE is not None:
        return _SECRETS_CACHE

    client = boto3.client("secretsmanager", region_name="us-east-1")

    voyage_arn = os.environ.get("SECRET_VOYAGE_ARN", "")
    gemini_arn = os.environ.get("SECRET_GEMINI_ARN", "")
    jina_arn = os.environ.get("SECRET_JINA_ARN", "")

    voyage_key = _resolve_secret(voyage_arn, client)
    gemini_key = _resolve_secret(gemini_arn, client)
    jina_key = _resolve_secret(jina_arn, client)

    # Fallback to env vars if secret resolution failed (dev/local mode)
    _SECRETS_CACHE = {
        "VOYAGE_API_KEY": voyage_key or os.environ.get("VOYAGE_API_KEY", ""),
        "GEMINI_API_KEY": gemini_key or os.environ.get("GEMINI_API_KEY", ""),
        "JINA_API_KEY": jina_key or os.environ.get("JINA_API_KEY", ""),
    }

    resolved = sum(1 for v in _SECRETS_CACHE.values() if v)
    logger.info("Resolved %d/%d secrets from Secrets Manager", resolved, len(_SECRETS_CACHE))

    return _SECRETS_CACHE


def inject_env() -> None:
    """Resolve secrets and inject them into os.environ.

    Call this at module load time (before any downstream code that reads env
    vars like VOYAGE_API_KEY). This is idempotent — the Secrets Manager call
    only happens on the first invocation.
    """
    secrets = get_secrets()
    for key, value in secrets.items():
        if value and not os.environ.get(key):
            os.environ[key] = value
