"""
leaf_validator.py — Two-Stage Leaf Validation for AgriGuard AI
===============================================================

Stage 1 — validate_image_quality(image)
    Pure-Python quality gating: blur, brightness, resolution.
    No API call. Always fast.

Stage 2 — validate_tomato_leaf(image)
    Gemini Vision check using the image's actual pixel data.
    Falls back to True (permissive) when Gemini is unavailable
    so that Groq-only / offline setups still work.

Public API
----------
validate_image_quality(image: PIL.Image) -> tuple[bool, str]
validate_tomato_leaf(image: PIL.Image)   -> tuple[bool, str]
"""

import io
import logging
import os

from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (tuned for outdoor leaf photography)
# ---------------------------------------------------------------------------
_MIN_RESOLUTION   = (100, 100)   # reject images smaller than 100×100
_BLUR_THRESHOLD   = 8.0          # edge std-dev below this → too blurry
_MIN_BRIGHTNESS   = 30           # mean pixel value below this → too dark
_MAX_BRIGHTNESS   = 240          # mean pixel value above this → over-exposed

# ---------------------------------------------------------------------------
# Gemini Vision prompt — structured 4-label response
# ---------------------------------------------------------------------------
_VALIDATION_PROMPT = """\
You are an agricultural image validator.

Determine whether the uploaded image contains a tomato leaf.

Respond ONLY with one of:

TOMATO_LEAF
OTHER_PLANT
NOT_A_PLANT
UNCLEAR"""

# Human-readable rejection messages for each non-tomato label
_REJECTION_MESSAGES = {
    "OTHER_PLANT": (
        "The image appears to contain a plant, but it is not a tomato leaf. "
        "Please upload a photograph of a tomato plant leaf."
    ),
    "NOT_A_PLANT": (
        "The image does not appear to contain a plant. "
        "Please upload a clear photograph of a real tomato leaf."
    ),
    "UNCLEAR": (
        "The image is unclear or ambiguous. "
        "Please upload a well-lit, in-focus photograph of a tomato leaf "
        "where the leaf occupies most of the frame."
    ),
}


# ---------------------------------------------------------------------------
# Stage 1: Image Quality Validation (no API)
# ---------------------------------------------------------------------------
def validate_image_quality(image: Image.Image) -> tuple[bool, str]:
    """
    Check whether *image* meets minimum quality requirements.

    Returns
    -------
    (True, "")                    — image is acceptable
    (False, human_readable_reason) — image should be rejected
    """
    # ── Resolution ──────────────────────────────────────────────────────────
    w, h = image.size
    min_w, min_h = _MIN_RESOLUTION
    if w < min_w or h < min_h:
        return (
            False,
            f"Image resolution is too low ({w}×{h} px). "
            f"Please upload an image of at least {min_w}×{min_h} pixels.",
        )

    # ── Brightness ───────────────────────────────────────────────────────────
    grayscale = image.convert("L")
    pixels    = list(grayscale.getdata())
    mean_val  = sum(pixels) / len(pixels)

    if mean_val < _MIN_BRIGHTNESS:
        return (
            False,
            "The image is too dark. Please use a well-lit photograph of the leaf.",
        )
    if mean_val > _MAX_BRIGHTNESS:
        return (
            False,
            "The image is over-exposed (too bright). "
            "Please photograph the leaf without harsh direct light.",
        )

    # ── Blur ─────────────────────────────────────────────────────────────────
    edges     = grayscale.filter(ImageFilter.FIND_EDGES)
    edge_data = list(edges.getdata())
    if len(edge_data) == 0:
        return False, "Could not assess image sharpness."

    edge_mean  = sum(edge_data) / len(edge_data)
    edge_sq    = sum((v - edge_mean) ** 2 for v in edge_data) / len(edge_data)
    edge_std   = edge_sq ** 0.5

    if edge_std < _BLUR_THRESHOLD:
        return (
            False,
            "The image appears blurry. "
            "Please hold the camera steady and ensure the leaf is in sharp focus.",
        )

    return True, ""


# ---------------------------------------------------------------------------
# Stage 2: Tomato Leaf Validation via Gemini Vision
# ---------------------------------------------------------------------------
def validate_tomato_leaf(image: Image.Image) -> tuple[bool, str]:
    """
    Ask Gemini Vision whether *image* shows a real tomato leaf.

    Returns
    -------
    (True,  "")              — Gemini confirmed a tomato leaf
    (True,  "skipped")       — Gemini unavailable; permissive pass-through
    (False, reason_string)   — Gemini rejected the image
    """
    # ── Try to get the Gemini client ─────────────────────────────────────────
    gemini_client = _get_gemini_client()
    if gemini_client is None:
        logger.info(
            "leaf_validator: Gemini unavailable — skipping visual validation."
        )
        return True, "skipped"

    # ── Convert image to JPEG bytes ──────────────────────────────────────────
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=85)
    image_bytes = buf.getvalue()

    # ── Call Gemini Vision ───────────────────────────────────────────────────
    try:
        from google.genai import types as _genai_types

        gemini_model = os.getenv("GEMINI_VISION_MODEL", "gemini-2.5-flash")


        response = gemini_client.models.generate_content(
            model=gemini_model,
            contents=[
                _genai_types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg",
                ),
                _genai_types.Part.from_text(_VALIDATION_PROMPT),
            ],
        )

        raw_text = (response.text or "").strip().upper()
        logger.info("Gemini Vision validation response: %r", raw_text)
        print(f"[LEAF VALIDATOR] Gemini Vision label: {raw_text!r}")

        # Extract the first recognised label from the response
        label = _parse_label(raw_text)

        if label == "TOMATO_LEAF":
            return True, ""

        # Any other label is a rejection — return the matching message
        rejection_msg = _REJECTION_MESSAGES.get(
            label,
            "This does not appear to be a tomato leaf. "
            "Please upload a clear photograph of a real tomato plant leaf.",
        )
        return False, rejection_msg

    except Exception as exc:  # noqa: BLE001
        logger.warning("Gemini Vision validation error: %s", exc)
        print(f"[LEAF VALIDATOR] Gemini Vision error: {exc}")
        # Fail open — don't block users due to transient API errors
        return True, "skipped"


# ---------------------------------------------------------------------------
# Internal helper: parse the 4-label response from Gemini
# ---------------------------------------------------------------------------
_KNOWN_LABELS = ("TOMATO_LEAF", "OTHER_PLANT", "NOT_A_PLANT", "UNCLEAR")


def _parse_label(raw_text: str) -> str:
    """
    Scan *raw_text* (already upper-cased) for the first recognised label.

    Returns one of: TOMATO_LEAF | OTHER_PLANT | NOT_A_PLANT | UNCLEAR.
    Falls back to UNCLEAR if the model produces unexpected output.
    """
    for label in _KNOWN_LABELS:
        if label in raw_text:
            return label
    logger.warning("Unrecognised Gemini label in response: %r — treating as UNCLEAR", raw_text)
    return "UNCLEAR"


# ---------------------------------------------------------------------------
# Internal helper: obtain the shared Gemini client
# ---------------------------------------------------------------------------
def _get_gemini_client():
    """
    Return the Gemini client initialised in llm_manager, or None.

    Accessing private module variables is intentional here — we want to
    reuse the already-authenticated client rather than duplicate setup.
    """
    try:
        import llm_manager as _lm
        if _lm._gemini_available and _lm._gemini_client is not None:
            return _lm._gemini_client
        return None
    except Exception:  # noqa: BLE001
        return None
