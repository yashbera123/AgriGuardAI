"""
llm_manager.py — Multi-Provider LLM Manager for AgriGuard AI
=============================================================

Provider priority order
-----------------------
1. Gemini 2.5 Flash   (GEMINI_API_KEY)
2. Groq Llama 3.3 70B (GROQ_API_KEY)
3. Local Knowledge Base (always available, zero-dependency fallback)

Keys are loaded from .env automatically via python-dotenv.

Every call returns a ``(response_text, provider_name)`` tuple so the UI
can display which provider answered.  Log lines are written to both the
Python logger and stdout so they appear in the Streamlit terminal:

    USING GEMINI
    USING GROQ
    USING LOCAL FALLBACK
"""

import logging
import os

# ---------------------------------------------------------------------------
# Load .env before reading any env vars
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass  # Rely on shell environment if python-dotenv is missing

# ---------------------------------------------------------------------------
# Streamlit Cloud secrets fallback
# On Streamlit Cloud, API keys are stored in st.secrets (not .env).
# We inject them into os.environ so the rest of this module works unchanged.
# ---------------------------------------------------------------------------
try:
    import streamlit as _st  # type: ignore
    _secrets = _st.secrets
    for _key in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        if _key not in os.environ or not os.environ[_key].strip():
            _val = _secrets.get(_key, "")
            if _val:
                os.environ[_key] = _val
except Exception:  # noqa: BLE001
    pass  # Not running inside Streamlit — ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider constants
# ---------------------------------------------------------------------------
PROVIDER_GEMINI = "Gemini 2.5 Flash"
PROVIDER_GROQ   = "Groq Llama 3.3 70B"
PROVIDER_LOCAL  = "Local Knowledge Base"

_GEMINI_MODEL = "gemini-2.5-flash"
_GROQ_MODEL   = "llama-3.3-70b-versatile"

# ---------------------------------------------------------------------------
# API key detection helpers
# ---------------------------------------------------------------------------
_PLACEHOLDER_VALUES = {"", "your_key_here", "your_gemini_api_key_here", "your_groq_api_key_here"}


def _valid_key(raw: str) -> bool:
    """Return True when *raw* looks like a real API key (not empty/placeholder)."""
    return bool(raw) and raw.strip() not in _PLACEHOLDER_VALUES


# ---------------------------------------------------------------------------
# Gemini initialisation
# ---------------------------------------------------------------------------
_gemini_client     = None
_gemini_available  = False
_gemini_init_error = ""

_raw_gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

print("=" * 55)
print("LLM MANAGER — PROVIDER INIT")
print("=" * 55)

if _valid_key(_raw_gemini_key):
    try:
        from google import genai as _google_genai  # type: ignore
        _gemini_client = _google_genai.Client(api_key=_raw_gemini_key)

        # Quick smoke-test
        _test_resp = _gemini_client.models.generate_content(
            model=_GEMINI_MODEL,
            contents="Reply with exactly one word: CONNECTED",
        )
        _test_text = (_test_resp.text or "").strip()
        if _test_text:
            _gemini_available = True
            print(f"[OK]  Gemini:  {PROVIDER_GEMINI} — connected")
        else:
            _gemini_init_error = "Empty test response from Gemini"
            print(f"[WARN] Gemini: empty test response — will skip")

    except ImportError:
        _gemini_init_error = "google-genai not installed"
        print(f"[FAIL] Gemini: {_gemini_init_error}")
    except Exception as exc:  # noqa: BLE001
        _gemini_init_error = str(exc)
        print(f"[FAIL] Gemini: {exc}")
else:
    _gemini_init_error = "GEMINI_API_KEY missing or placeholder"
    print(f"[SKIP] Gemini: {_gemini_init_error}")

# ---------------------------------------------------------------------------
# Groq initialisation
# ---------------------------------------------------------------------------
_groq_client     = None
_groq_available  = False
_groq_init_error = ""

_raw_groq_key = os.getenv("GROQ_API_KEY", "").strip()

if _valid_key(_raw_groq_key):
    try:
        from groq import Groq as _GroqClient  # type: ignore
        _groq_client = _GroqClient(api_key=_raw_groq_key)

        # Quick smoke-test
        _groq_test = _groq_client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[{"role": "user", "content": "Reply with exactly one word: CONNECTED"}],
            max_tokens=10,
        )
        _groq_text = (_groq_test.choices[0].message.content or "").strip()
        if _groq_text:
            _groq_available = True
            print(f"[OK]  Groq:    {PROVIDER_GROQ} — connected")
        else:
            _groq_init_error = "Empty test response from Groq"
            print(f"[WARN] Groq:   empty test response — will skip")

    except ImportError:
        _groq_init_error = "groq package not installed (run: pip install groq)"
        print(f"[FAIL] Groq: {_groq_init_error}")
    except Exception as exc:  # noqa: BLE001
        _groq_init_error = str(exc)
        print(f"[FAIL] Groq: {exc}")
else:
    _groq_init_error = "GROQ_API_KEY missing or placeholder"
    print(f"[SKIP] Groq: {_groq_init_error}")

print("[OK]  Local KB always available as final fallback")
print("=" * 55)


# ---------------------------------------------------------------------------
# Runtime provider state (can change during a session if Gemini hits quota)
# ---------------------------------------------------------------------------
# Set initial provider to best available (not always LOCAL)
if _gemini_available:
    _current_provider: str = PROVIDER_GEMINI
elif _groq_available:
    _current_provider: str = PROVIDER_GROQ
else:
    _current_provider: str = PROVIDER_LOCAL


def get_current_provider() -> str:
    """Return the name of the provider that was used most recently."""
    return _current_provider


def get_provider_status() -> dict:
    """
    Return a summary dict for display in the Streamlit sidebar.

    Keys
    ----
    current_provider : str
    gemini_available : bool
    groq_available   : bool
    gemini_error     : str
    groq_error       : str
    """
    return {
        "current_provider": _current_provider,
        "gemini_available": _gemini_available,
        "groq_available":   _groq_available,
        "gemini_error":     _gemini_init_error,
        "groq_error":       _groq_init_error,
    }


# ---------------------------------------------------------------------------
# Internal: raw provider calls
# ---------------------------------------------------------------------------
def _call_gemini_raw(prompt: str) -> str | None:
    """
    Send *prompt* to Gemini 2.5 Flash.

    Returns response text on success, None on any failure (including quota
    exhausted — HTTP 429 / ResourceExhausted).
    """
    if not _gemini_available or _gemini_client is None:
        return None
    try:
        print("CALLING GEMINI...")
        response = _gemini_client.models.generate_content(
            model=_GEMINI_MODEL,
            contents=prompt,
        )
        text = (response.text or "").strip()
        if text:
            print(f"GEMINI RESPONSE RECEIVED ({len(text)} chars)")
        else:
            print("GEMINI RESPONSE: empty")
        return text or None
    except Exception as exc:  # noqa: BLE001
        exc_str = str(exc).lower()
        if "quota" in exc_str or "429" in exc_str or "resource_exhausted" in exc_str:
            print(f"GEMINI QUOTA EXHAUSTED — switching to Groq. Detail: {exc}")
        else:
            print(f"GEMINI ERROR: {exc}")
        logger.warning("Gemini call failed: %s", exc)
        return None


def _call_groq_raw(prompt: str, system_prompt: str = "") -> str | None:
    """
    Send *prompt* to Groq Llama 3.3 70B.

    Returns response text on success, None on any failure.
    """
    if not _groq_available or _groq_client is None:
        return None
    try:
        print("CALLING GROQ...")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        completion = _groq_client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=messages,
            max_tokens=2048,
            temperature=0.7,
        )
        text = (completion.choices[0].message.content or "").strip()
        if text:
            print(f"GROQ RESPONSE RECEIVED ({len(text)} chars)")
        else:
            print("GROQ RESPONSE: empty")
        return text or None
    except Exception as exc:  # noqa: BLE001
        print(f"GROQ ERROR: {exc}")
        logger.warning("Groq call failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public: generate_with_fallback
# ---------------------------------------------------------------------------
def generate_with_fallback(
    prompt: str,
    system_prompt: str = "",
    local_fallback_fn=None,
    local_fallback_args: dict | None = None,
) -> tuple[str, str]:
    """
    Try providers in priority order and return the first successful response.

    Priority:  Gemini  →  Groq  →  Local KB

    Parameters
    ----------
    prompt : str
        The user / task prompt to send to cloud providers.
    system_prompt : str, optional
        System-level instruction (used by Groq; prepended to Gemini prompt).
    local_fallback_fn : callable, optional
        Function to call for local KB response.  Receives ``**local_fallback_args``.
    local_fallback_args : dict, optional
        Keyword arguments forwarded to *local_fallback_fn*.

    Returns
    -------
    (response_text, provider_name) : tuple[str, str]
    """
    global _current_provider

    # ── 1. Gemini ────────────────────────────────────────────────────────────
    if _gemini_available:
        full_prompt = f"{system_prompt}\n\n{prompt}".strip() if system_prompt else prompt
        gemini_result = _call_gemini_raw(full_prompt)
        if gemini_result:
            _current_provider = PROVIDER_GEMINI
            logger.info("USING GEMINI")
            print("USING GEMINI")
            return gemini_result, PROVIDER_GEMINI

        # Gemini failed (quota or network) — cascade to Groq
        logger.warning("Gemini unavailable/exhausted — cascading to Groq")
        print("GEMINI UNAVAILABLE — CASCADING TO GROQ")

    # ── 2. Groq ──────────────────────────────────────────────────────────────
    if _groq_available:
        groq_result = _call_groq_raw(prompt, system_prompt=system_prompt)
        if groq_result:
            _current_provider = PROVIDER_GROQ
            logger.info("USING GROQ")
            print("USING GROQ")
            return groq_result, PROVIDER_GROQ

        logger.warning("Groq call failed — cascading to Local KB")
        print("GROQ UNAVAILABLE — CASCADING TO LOCAL KB")

    # ── 3. Local KB ──────────────────────────────────────────────────────────
    _current_provider = PROVIDER_LOCAL
    logger.info("USING LOCAL FALLBACK")
    print("USING LOCAL FALLBACK")

    if local_fallback_fn is not None:
        result = local_fallback_fn(**(local_fallback_args or {}))
        return str(result), PROVIDER_LOCAL

    return (
        "I'm currently unable to connect to AI services. "
        "Please check your API keys in .env and try again.",
        PROVIDER_LOCAL,
    )
