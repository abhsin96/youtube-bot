"""
Secure API key storage via the OS keyring.

Backend selection
-----------------
macOS   → Keychain
Windows → Windows Credential Manager
Linux   → Secret Service (requires dbus + a daemon such as gnome-keyring or
          kwallet).  Install with:

              sudo apt-get install gnome-keyring dbus-x11
              eval $(dbus-launch --sh-syntax)
              gnome-keyring-daemon --start --components=secrets

Headless Linux fallback
-----------------------
If no Secret Service daemon is available, ``keyring`` raises
``NoKeyringError``.  In that case the module falls back to a file stored at
``~/.config/youtube-rag-extension/creds``.  The file is protected with 0600
permissions and the value is base64-encoded to prevent casual inspection.

WARNING: base64 is *not* encryption.  On production headless Linux, prefer
setting ``OPENAI_API_KEY`` as an environment variable managed by your secret
manager (Vault, AWS SSM, etc.) rather than using this fallback.
"""

from __future__ import annotations

import base64
from pathlib import Path

import keyring
import keyring.errors
import structlog

logger = structlog.get_logger(__name__)

_SERVICE = "youtube-rag-extension"
_USER = "openai"
_FALLBACK_PATH = Path.home() / ".config" / "youtube-rag-extension" / "creds"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def set_api_key(key: str) -> None:
    """Persist *key* in the OS keyring (or the file-based fallback)."""
    try:
        keyring.set_password(_SERVICE, _USER, key)
    except keyring.errors.NoKeyringError:
        _warn_fallback("set")
        _fallback_write(key)


def get_api_key() -> str | None:
    """Return the stored key, or *None* if nothing has been stored."""
    try:
        return keyring.get_password(_SERVICE, _USER)
    except keyring.errors.NoKeyringError:
        _warn_fallback("get")
        return _fallback_read()


def clear_api_key() -> None:
    """Remove the stored key from whichever backend holds it."""
    try:
        keyring.delete_password(_SERVICE, _USER)
    except keyring.errors.PasswordDeleteError:
        pass  # already absent — treat as a no-op
    except keyring.errors.NoKeyringError:
        _fallback_clear()


def get_openai_key(settings) -> str | None:
    """Resolve the OpenAI key: keyring first, then ``settings.openai_api_key``.

    Returns *None* if no key is available from either source.
    """
    key = get_api_key()
    if key:
        return key
    fallback = getattr(settings, "openai_api_key", "") or ""
    return fallback or None


# ---------------------------------------------------------------------------
# File-based fallback (headless Linux)
# ---------------------------------------------------------------------------


def _warn_fallback(op: str) -> None:
    logger.warning(
        "OS keyring unavailable — using file-based fallback (less secure). "
        "On Linux install dbus + gnome-keyring, or set OPENAI_API_KEY env var.",
        operation=op,
        fallback_path=str(_FALLBACK_PATH),
    )


def _fallback_write(key: str) -> None:
    _FALLBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    _FALLBACK_PATH.write_bytes(base64.b64encode(key.encode()))
    _FALLBACK_PATH.chmod(0o600)


def _fallback_read() -> str | None:
    if not _FALLBACK_PATH.exists():
        return None
    try:
        return base64.b64decode(_FALLBACK_PATH.read_bytes()).decode()
    except Exception:
        logger.error("fallback credentials file is corrupted", path=str(_FALLBACK_PATH))
        return None


def _fallback_clear() -> None:
    if _FALLBACK_PATH.exists():
        _FALLBACK_PATH.unlink()
