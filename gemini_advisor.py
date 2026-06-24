"""
gemini_advisor.py — AI Agriculture Advisor for AgriGuard AI
============================================================

Public API (unchanged — app.py imports these directly):
    generate_advisory(disease, cause, symptoms, medication) -> dict
    chat_with_advisor(user_question, detected_disease, chat_history) -> str
    is_gemini_available() -> bool
    get_gemini_status()   -> str

Backend selection is now delegated to llm_manager.py:
    1. Gemini 2.5 Flash   (GEMINI_API_KEY in .env)
    2. Groq Llama 3.3 70B (GROQ_API_KEY   in .env)
    3. Local Knowledge Base (ai_advisor.py — zero-dependency fallback)

The module never crashes due to API failures; it silently cascades.
"""

import logging

# ---------------------------------------------------------------------------
# LLM Manager — multi-provider backend
# ---------------------------------------------------------------------------
from llm_manager import (  # noqa: E402
    PROVIDER_GEMINI,
    PROVIDER_GROQ,
    PROVIDER_LOCAL,
    generate_with_fallback,
    get_current_provider,
    get_provider_status,
)

# ---------------------------------------------------------------------------
# Fallback advisor — always available, no external dependency
# ---------------------------------------------------------------------------
from ai_advisor import (  # noqa: E402
    SAFETY_DISCLAIMER,
    chat_with_advisor as _fallback_chat,
    generate_advisory as _fallback_advisory,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers (backward-compatible with existing app.py calls)
# ---------------------------------------------------------------------------
def is_gemini_available() -> bool:
    """Return True if Gemini is the currently active provider."""
    status = get_provider_status()
    return status["gemini_available"]


def is_groq_available() -> bool:
    """Return True if Groq is available (may serve as active fallback)."""
    status = get_provider_status()
    return status["groq_available"]


def get_gemini_status() -> str:
    """
    Return a human-readable status string for display in the UI.

    Kept for backward-compatibility with existing app.py calls.
    """
    provider = get_current_provider()
    if provider == PROVIDER_GEMINI:
        return " Gemini 2.5 Flash Connected"
    if provider == PROVIDER_GROQ:
        return " Groq Llama 3.3 70B Active (Gemini fallback)"
    return " Using Local Knowledge Base (no cloud provider available)"


# ---------------------------------------------------------------------------
# Advisory prompt
# ---------------------------------------------------------------------------
_ADVISORY_SYSTEM = (
    "You are an expert agricultural advisor specialising in tomato crop diseases. "
    "Always use practical, farmer-friendly language."
)

_ADVISORY_PROMPT = """\
A tomato plant has been diagnosed with the following condition.

Disease: {disease_display}

Cause:
{cause}

Symptoms:
{symptoms}

Suggested Medication:
{medication}

Provide a comprehensive response with exactly these five numbered sections.
Use the exact headings shown. Keep language practical and easy for farmers to understand.

1. Simple Explanation for Farmers
   Explain what this disease is, why it happens, and how serious it is in plain language.

2. Preventive Measures
   List 4-5 concrete, actionable steps a farmer can take to prevent this disease.

3. Sustainable Farming Recommendations
   Provide eco-friendly, low-chemical approaches that reduce environmental impact.

4. Treatment Strategy
   Give a clear step-by-step treatment plan for managing an active infection.

5. Long-Term Crop Protection Advice
   Describe season-over-season strategies to protect the crop in future.

End your response with this exact line:
 Consult local agricultural experts before applying chemicals or changing farming practices.
"""

# ---------------------------------------------------------------------------
# Chatbot system prompt
# ---------------------------------------------------------------------------
_CHAT_SYSTEM = (
    "You are AgriGuard AI, an intelligent assistant for tomato crop disease management "
    "and sustainable farming. You can answer any question — including general knowledge "
    "questions, greetings, and farming questions. "
    "When the user asks about a specific detected disease, use that context. "
    "When asked general questions (who are you, capital cities, weather, etc.) answer them "
    "directly and naturally. "
    "For responses that involve chemicals or farming practice changes, always end with:\n"
    " Consult local agricultural experts before applying chemicals or changing farming practices."
)


# ---------------------------------------------------------------------------
# Internal: parse advisory text → structured dict
# ---------------------------------------------------------------------------
def _parse_advisory_text(raw_text: str) -> dict:
    """
    Split a five-section advisory response into the advisory dict.

    Falls back to placing the full text in farmer_explanation if nothing matches.
    """
    sections = {
        "farmer_explanation": "",
        "prevention": "",
        "sustainability": "",
        "treatment_strategy": "",
        "long_term_advice": "",
    }

    markers = [
        ("farmer_explanation", ["simple explanation", "explanation for farmer", "1."]),
        ("prevention",         ["preventive measure", "prevention", "2."]),
        ("sustainability",     ["sustainable farming", "sustainability", "eco-friendly", "3."]),
        ("treatment_strategy", ["treatment strategy", "treatment plan", "4."]),
        ("long_term_advice",   ["long-term", "long term", "crop protection advice", "5."]),
    ]

    current_key = None
    buffer: list[str] = []

    def _flush(key, buf):
        if key and buf:
            sections[key] = "\n".join(buf).strip()

    for line in raw_text.splitlines():
        line_lower = line.lower()
        matched = False
        for key, keywords in markers:
            if any(kw in line_lower for kw in keywords):
                _flush(current_key, buffer)
                current_key = key
                buffer = []
                matched = True
                break
        if not matched:
            buffer.append(line)

    _flush(current_key, buffer)

    # Nothing was parsed — put entire text in farmer_explanation
    if not any(sections.values()):
        sections["farmer_explanation"] = raw_text

    # Guarantee disclaimer in every populated field
    for key in sections:
        if sections[key] and SAFETY_DISCLAIMER not in sections[key]:
            sections[key] = f"{sections[key]}\n\n{SAFETY_DISCLAIMER}"

    return sections


# ---------------------------------------------------------------------------
# PUBLIC API: generate_advisory
# ---------------------------------------------------------------------------
def generate_advisory(
    disease: str,
    cause: str = "",
    symptoms: str = "",
    medication: str = "",
) -> dict:
    """
    Generate a structured AI advisory for a detected disease.

    Tries providers in order: Gemini → Groq → Local KB.

    Returns
    -------
    dict
        Keys: farmer_explanation, prevention, sustainability,
              treatment_strategy, long_term_advice, source, provider
        provider — human-readable provider name (e.g. "Gemini 2.5 Flash")
        source   — "gemini" | "groq" | "local"  (legacy compat key)
    """
    disease_display = disease.replace("Tomato_", "Tomato ").replace("_", " ")

    prompt = _ADVISORY_PROMPT.format(
        disease_display=disease_display,
        cause=cause.strip() or "Not specified.",
        symptoms=symptoms.strip() or "Not specified.",
        medication=medication.strip() or "Not specified.",
    )

    raw, provider = generate_with_fallback(
        prompt=prompt,
        system_prompt=_ADVISORY_SYSTEM,
        local_fallback_fn=_fallback_advisory,
        local_fallback_args={
            "disease": disease,
            "cause": cause,
            "symptoms": symptoms,
            "medication": medication,
        },
    )

    # Determine legacy source key
    if provider == PROVIDER_GEMINI:
        source = "gemini"
    elif provider == PROVIDER_GROQ:
        source = "groq"
    else:
        source = "local"

    # Local KB returns a ready-made dict; cloud providers return raw text
    if source == "local":
        result = raw if isinstance(raw, dict) else _fallback_advisory(disease, cause, symptoms, medication)
        if not isinstance(result, dict):
            result = _fallback_advisory(disease, cause, symptoms, medication)
        result["source"] = source
        result["provider"] = provider
        return result

    # Parse the cloud response
    parsed = _parse_advisory_text(raw)
    parsed["source"] = source
    parsed["provider"] = provider
    return parsed


# ---------------------------------------------------------------------------
# PUBLIC API: chat_with_advisor
# ---------------------------------------------------------------------------
def chat_with_advisor(
    user_question: str,
    detected_disease: str = "",
    chat_history: list = None,
) -> str:
    """
    Generate a chatbot response.

    Tries providers in order: Gemini → Groq → Local KB.

    Parameters
    ----------
    user_question : str
        The farmer's question in plain text.
    detected_disease : str, optional
        Internal class name of last detected disease.
    chat_history : list, optional
        Previous {\"role\": ..., \"content\": ...} entries.

    Returns
    -------
    str
        Response string with safety disclaimer appended where relevant.
    """
    # -- Build disease context block -----------------------------------------
    disease_context_block = ""
    if detected_disease:
        disease_display = detected_disease.replace("Tomato_", "Tomato ").replace("_", " ")
        disease_context_block = (
            f"\n\nFor context: the farmer's tomato crop has recently been diagnosed "
            f"with {disease_display}. Use this context ONLY if the question relates "
            f"to this disease or farming. Do NOT force disease content into "
            f"unrelated questions."
        )

    # -- Include recent conversation history (last 6 turns) ------------------
    history_block = ""
    if chat_history:
        recent = chat_history[-6:]
        lines = []
        for msg in recent:
            role = "Farmer" if msg["role"] == "user" else "AgriGuard AI"
            lines.append(f"{role}: {msg['content']}")
        history_block = "\n\nRecent conversation:\n" + "\n".join(lines)

    # -- Assemble user prompt ------------------------------------------------
    prompt = (
        f"{disease_context_block}"
        f"{history_block}"
        f"\n\nFarmer's question: {user_question}"
    ).strip()

    raw, _provider = generate_with_fallback(
        prompt=prompt,
        system_prompt=_CHAT_SYSTEM,
        local_fallback_fn=_fallback_chat,
        local_fallback_args={
            "user_question": user_question,
            "detected_disease": detected_disease,
            "chat_history": chat_history,
        },
    )

    # For local KB the return is already a ready-made string
    if not isinstance(raw, str):
        raw = str(raw)

    # Add disclaimer only if the response involves chemicals/practices
    if SAFETY_DISCLAIMER not in raw:
        trigger_words = [
            "fungicide", "pesticide", "chemical", "spray", "apply",
            "treatment", "fertiliser", "fertilizer", "practice", "herbicide",
            "insecticide", "bactericide",
        ]
        if any(w in raw.lower() for w in trigger_words):
            raw = f"{raw}\n\n{SAFETY_DISCLAIMER}"

    return raw
