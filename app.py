"""
app.py — AgriGuard AI  v2.0
============================

AI-Powered Sustainable Agriculture Decision Support System.

Stack:
  TensorFlow MobileNetV2  — disease classification
  Gemini 2.5 Flash        — vision validation + AI advisor + chatbot
  Groq Llama 3.3 70B      — automatic advisor/chatbot fallback
  Local Knowledge Base    — always-available offline fallback
"""

import hashlib
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from analytics import render_analytics_dashboard
from crop_knowledge import ACTIVE_CROP, crop_knowledge, get_crop_config
from disease_info import disease_info
from leaf_validator import validate_image_quality, validate_tomato_leaf
from pdf_generator import generate_pdf_report
from prediction_logger import append_prediction, load_history
from recommendations import recommendations

from gemini_advisor import (
    chat_with_advisor,
    generate_advisory,
    get_gemini_status,
    is_gemini_available,
    is_groq_available,
)
from llm_manager import (
    PROVIDER_GEMINI,
    PROVIDER_GROQ,
    PROVIDER_LOCAL,
    get_current_provider,
    get_provider_status,
)


# ---------------------------------------------------------------------------
# Validation confidence threshold
# ---------------------------------------------------------------------------
# Only extremely high-confidence predictions (>=95%) bypass Gemini Vision.
# Everything below always goes through Gemini validation — a score below 95%
# is NOT a reliable indicator the image is a real tomato leaf.
CONFIDENCE_THRESHOLD = 95.0



def _init_validation_counters():
    """Ensure session-state counters exist."""
    if "gemini_validation_calls" not in st.session_state:
        st.session_state["gemini_validation_calls"] = 0
    if "local_validation_passes" not in st.session_state:
        st.session_state["local_validation_passes"] = 0


st.set_page_config(
    page_title="AgriGuard AI — Sustainable Agriculture Platform",
    page_icon="🌱",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Minimal CSS — only fonts and progress-bar polish (no layout HTML)
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
        [data-testid="stProgress"] > div > div {
            border-radius: 999px;
            background: linear-gradient(90deg, #22c55e, #4ade80) !important;
        }
        [data-testid="stExpander"] summary { font-weight: 600; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def display_name_for(class_name, crop_config):
    return crop_config["display_names"].get(class_name, class_name.replace("_", " "))


def load_leaf_image(image_file):
    if image_file is None:
        return None, None
    file_bytes = image_file.getvalue()
    if not file_bytes:
        return None, None
    try:
        image = Image.open(BytesIO(file_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        st.error("The selected image could not be read. Please try another image.")
        return None, None
    fingerprint = hashlib.sha256(file_bytes).hexdigest()
    return image, fingerprint


def predict_disease(image, trained_model, crop_config):
    class_names = crop_config["class_names"]
    image_size = crop_config.get("image_size", (224, 224))
    resized = image.resize(image_size)
    arr = np.expand_dims(np.array(resized), axis=0)
    arr = preprocess_input(arr)
    prediction = trained_model.predict(arr, verbose=0)
    probs = prediction[0].astype(float)
    if len(probs) != len(class_names):
        raise ValueError("Model output size does not match the configured disease classes.")
    top_idx = np.argsort(probs)[::-1][:5]
    return {
        "predicted_class": class_names[int(top_idx[0])],
        "confidence": float(probs[int(top_idx[0])] * 100),
        "probabilities": probs,
        "top_indices": top_idx,
    }


def build_probability_table(probabilities, crop_config):
    class_names = crop_config["class_names"]
    return pd.DataFrame({
        "Disease": [display_name_for(c, crop_config) for c in class_names],
        "Probability (%)": [round(float(p) * 100, 2) for p in probabilities],
    })


def build_top_prediction_table(result, crop_config):
    class_names = crop_config["class_names"]
    rows = []
    for idx in result["top_indices"]:
        rows.append({
            "Disease": display_name_for(class_names[int(idx)], crop_config),
            "Confidence": round(float(result["probabilities"][int(idx)] * 100), 2),
        })
    return pd.DataFrame(rows)


def get_prediction_context(result, crop_config):
    pc = result["predicted_class"]
    display_name = display_name_for(pc, crop_config)
    sustainability_score = crop_config["sustainability_scores"].get(pc, 60)
    knowledge = crop_knowledge.get(pc, {})
    disease_data = recommendations.get(pc, {})
    return {
        "predicted_class": pc,
        "display_name": display_name,
        "confidence": result["confidence"],
        "sustainability_score": sustainability_score,
        "description": disease_info.get(pc, "Information unavailable."),
        "cause": knowledge.get("cause", "Information unavailable."),
        "symptoms": knowledge.get("symptoms", "Information unavailable."),
        "medication": knowledge.get("medication", "Information unavailable."),
        "severity": disease_data.get("severity", "Unknown"),
        "treatment": disease_data.get("treatment", "Information unavailable."),
        "sustainability": disease_data.get("sustainability", "Information unavailable."),
        "prevention": disease_data.get("prevention", "Information unavailable."),
    }


def log_prediction_once(history_df, image_fingerprint, context):
    if "logged_prediction_keys" not in st.session_state:
        st.session_state["logged_prediction_keys"] = set()
    log_key = f"{image_fingerprint}:{context['predicted_class']}:{context['confidence']:.2f}"
    if log_key in st.session_state["logged_prediction_keys"]:
        return history_df
    try:
        updated = append_prediction(
            disease_name=context["display_name"],
            confidence=context["confidence"],
            sustainability_score=context["sustainability_score"],
        )
        st.session_state["logged_prediction_keys"].add(log_key)
        return updated
    except Exception as exc:
        st.warning(f"Prediction history could not be updated: {exc}")
        return history_df


def build_report_data(context, image_fingerprint, advisor_data=None):
    now = datetime.now()
    report_id = f"AGR-{now.strftime('%Y%m%d%H%M%S')}-{image_fingerprint[:6].upper()}"
    report = {
        "report_id": report_id,
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "disease_name": context["display_name"],
        "confidence_score": f"{context['confidence']:.2f}%",
        "cause": context["cause"],
        "symptoms": context["symptoms"],
        "medication": context["medication"],
        "treatment": context["treatment"],
        "sustainability_advice": context["sustainability"],
        "sustainability_score": f"{context['sustainability_score']}/100",
        "prevention_tips": context["prevention"],
    }
    if advisor_data:
        report["advisor_data"] = advisor_data
    return report


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(crop_config):
    status = get_provider_status()
    current = get_current_provider()

    with st.sidebar:
        st.title("🌱 AgriGuard AI")
        st.caption("v2.0 — Sustainable Agriculture Platform")
        st.divider()

        st.subheader("📋 Project Overview")
        st.metric("Model",   crop_config.get("model_name", "MobileNetV2"))
        st.metric("Crop",    crop_config.get("crop_name",  "Tomato"))
        st.metric("AI",      "Gemini + Groq")
        st.metric("Version", "2.0")

        st.divider()

        st.subheader("🤖 AI Provider Status")
        if current == PROVIDER_GEMINI:
            st.success(f"Active: {PROVIDER_GEMINI}")
        elif current == PROVIDER_GROQ:
            st.warning(f"Active: {PROVIDER_GROQ}")
        else:
            st.error(f"Active: {PROVIDER_LOCAL}")

        if status["gemini_available"]:
            st.write("🟢 Gemini 2.5 Flash — ready")
        else:
            st.write(f"🔴 Gemini 2.5 Flash — unavailable")
            st.caption(status["gemini_error"])

        if status["groq_available"]:
            st.write("🟢 Groq Llama 3.3 70B — ready")
        else:
            st.write("🔴 Groq Llama 3.3 70B — unavailable")
            st.caption(status["groq_error"])

        st.write("🟢 Local Knowledge Base — always ready")

        st.divider()
        st.caption("Priority: Gemini → Groq → Local KB. Keys loaded from `.env`.")

        # ── Validation Analytics ─────────────────────────────────────────
        st.divider()
        st.subheader("🔍 Validation Analytics")
        _init_validation_counters()
        gcalls = st.session_state.get("gemini_validation_calls", 0)
        lpasses = st.session_state.get("local_validation_passes", 0)
        total = gcalls + lpasses
        st.metric("Local Validation Passes",  lpasses,
                  help="Images that skipped Gemini Vision (confidence ≥95%)")
        st.metric("Gemini Validation Calls",   gcalls,
                  help="Images that triggered Gemini Vision (confidence <95%)")
        if total > 0:
            saved_pct = round(lpasses / total * 100)
            st.progress(lpasses / total)
            st.caption(f"**{saved_pct}%** of images validated locally — {lpasses} Gemini call(s) saved")
        else:
            st.caption("No images validated yet in this session.")

        st.divider()
        st.subheader("🏆 Built For")
        st.write("✅ 1M1B AI for Sustainability")
        st.write("✅ College Major Project")
        st.write("✅ Placement Portfolio")


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
def render_header(crop_config):
    st.title("🌱 AgriGuard AI")
    st.subheader("AI-Powered Sustainable Agriculture Decision Support System")

    # Tech badges as a single caption line
    st.caption("✓ Computer Vision   ·   ✓ Gemini AI   ·   ✓ Sustainability   ·   ✓ Real-Time Insights")

    # Provider status
    active_provider = get_current_provider()
    status_text = get_gemini_status()
    if active_provider == PROVIDER_GEMINI:
        st.success(f"**{status_text}** — AI Advisor powered by Gemini 2.5 Flash")
    elif active_provider == PROVIDER_GROQ:
        st.warning(f"**{status_text}** — Using Groq Llama 3.3 70B (Gemini fallback)")
    else:
        st.warning(
            f"**{status_text}**  \n"
            "Add `GEMINI_API_KEY` or `GROQ_API_KEY` in `.env` to enable cloud AI."
        )

    st.divider()

    # Stat metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Disease Classes", len(crop_config["class_names"]))
    col2.metric("Model", crop_config["model_name"])
    col3.metric("Crop", crop_config["crop_name"])
    col4.metric("AI Advisor", active_provider)

    st.divider()


# ---------------------------------------------------------------------------
# Image source selector
# ---------------------------------------------------------------------------
def render_image_source_selector():
    st.subheader("📷 Image Capture")
    st.caption("Upload a tomato leaf photo or use your camera for instant AI diagnosis.")

    source = st.radio(
        "Image source",
        ["📁 Upload Image", "📸 Capture from Camera"],
        horizontal=True,
        label_visibility="collapsed",
    )

    input_col, tip_col = st.columns([1.4, 1])

    with input_col:
        if source == "📁 Upload Image":
            image_file = st.file_uploader(
                "Upload Tomato Leaf Image",
                type=["jpg", "jpeg", "png"],
            )
            source_label = "Uploaded"
        else:
            image_file = st.camera_input("Capture Tomato Leaf Image")
            source_label = "Captured"

    with tip_col:
        st.info(
            "**💡 For best results**\n\n"
            "- Use natural or bright indoor light\n"
            "- Leaf should fill most of the frame\n"
            "- Avoid blurry or very dark images\n"
            "- Tomato leaves only — other plants are rejected"
        )

    return image_file, source_label


# ---------------------------------------------------------------------------
# Validation — confidence-gated Gemini Vision
# ---------------------------------------------------------------------------
def render_validation_error(message: str):
    st.error(
        f"\u274c **Invalid Image**\n\n{message}\n\n"
        "Please upload a clear, well-lit photograph of a **real tomato leaf**."
    )


def run_validation(
    image: Image.Image,
    image_fingerprint: str,
    confidence: float,
) -> tuple[bool, str]:
    """
    Two-stage confidence-gated validation.

    Stage 1 — Local quality check (always, no API call).
    Stage 2 — Decision engine:
        If confidence >= CONFIDENCE_THRESHOLD  →  skip Gemini (local pass).
        If confidence <  CONFIDENCE_THRESHOLD  →  call Gemini Vision.

    Returns (validation_passed: bool, source: str)
        source is one of: "local" | "gemini" | "gemini_fallback"
    """
    _init_validation_counters()

    # ── Stage 1: Image quality ────────────────────────────────────────────
    quality_ok, quality_msg = validate_image_quality(image)
    if not quality_ok:
        render_validation_error(quality_msg)
        st.stop()

    # ── Stage 2: Confidence-gated decision ───────────────────────────────
    if confidence >= CONFIDENCE_THRESHOLD:
        # High-confidence → trust the model, skip Gemini Vision
        st.session_state["local_validation_passes"] += 1
        return True, "local"

    # Low-confidence → use Gemini Vision (with session-state cache)
    cached_hash   = st.session_state.get("validated_image_hash")
    cached_result = st.session_state.get("validation_result")

    if cached_hash == image_fingerprint and cached_result is not None:
        # Cache hit — reuse previous Gemini result (no extra API call)
        leaf_valid, leaf_msg, source = cached_result
    else:
        # Cache miss — call Gemini Vision
        with st.spinner("\U0001f50d Low confidence detected — verifying with Gemini Vision..."):
            leaf_valid, leaf_msg = validate_tomato_leaf(image)

        source = "gemini"
        if leaf_msg == "skipped":
            # Gemini unavailable — fail open
            leaf_valid = True
            source = "gemini_fallback"

        # Increment call counter only for genuine new API calls
        if source == "gemini":
            st.session_state["gemini_validation_calls"] += 1

        # Cache result for this image fingerprint
        st.session_state["validated_image_hash"] = image_fingerprint
        st.session_state["validation_result"]    = (leaf_valid, leaf_msg, source)

    if not leaf_valid:
        render_validation_error(leaf_msg)
        st.stop()

    return True, source


# ---------------------------------------------------------------------------
# Prediction results — all native Streamlit components
# ---------------------------------------------------------------------------
def render_prediction_results(image, source_label, result, context, crop_config, image_fingerprint):

    st.divider()
    st.subheader("🔬 Diagnosis Results")

    # ── Image + Result summary ──────────────────────────────────────────────
    img_col, res_col = st.columns([1, 1])

    with img_col:
        st.image(image, caption=f"{source_label} Leaf Image", use_container_width=True)

    with res_col:
        severity = context["severity"]
        disease  = context["display_name"]
        conf     = context["confidence"]
        sust     = context["sustainability_score"]

        # Disease + severity
        if severity == "High":
            st.error(f"### 🦠 {disease}")
            st.error(f"**Severity:** {severity}")
        elif severity == "Medium":
            st.warning(f"### 🦠 {disease}")
            st.warning(f"**Severity:** {severity}")
        elif severity in ("None", "Low"):
            st.success(f"### 🌿 {disease}")
            st.success(f"**Severity:** {severity}")
        else:
            st.info(f"### 🦠 {disease}")
            st.info(f"**Severity:** {severity}")

        st.write("")

        # Confidence
        st.write("**Confidence Score**")
        st.progress(min(conf / 100, 1.0))
        st.caption(f"{conf:.1f}%")

        # Sustainability
        st.write("**Sustainability Score**")
        st.progress(sust / 100)
        st.caption(f"{sust}/100")

        # Quick metrics
        mc1, mc2 = st.columns(2)
        mc1.metric("Confidence",    f"{conf:.1f}%")
        mc2.metric("Sustainability", f"{sust}/100")

        # Validation source badge (injected by caller via session state)
        vsource = st.session_state.get("last_validation_source", "")
        if vsource == "local":
            st.success("✅ Validation Source: Local Model  (Gemini Vision skipped)")
        elif vsource == "gemini":
            st.info("🤖 Validation Source: Gemini Vision  (low-confidence image)")
        elif vsource == "gemini_fallback":
            st.warning("⚠️ AI validation unavailable.  Proceeding with model prediction.")

    # ── Top 5 Predictions ──────────────────────────────────────────────────
    st.divider()
    top_table = build_top_prediction_table(result, crop_config)
    chart_col, bar_col = st.columns(2)

    with chart_col:
        st.subheader("🏆 Top 5 Predictions")
        max_conf = top_table["Confidence"].max() or 1
        for _, row in top_table.iterrows():
            st.write(f"**{row['Disease']}** — {row['Confidence']:.1f}%")
            st.progress(row["Confidence"] / 100)

    with bar_col:
        st.subheader("📊 Confidence Comparison")
        st.bar_chart(top_table.set_index("Disease"))

    with st.expander("📋 Full Probability Table"):
        prob_table = build_probability_table(result["probabilities"], crop_config)
        st.dataframe(
            prob_table.sort_values(by="Probability (%)", ascending=False),
            use_container_width=True,
        )

    # ── Disease details ────────────────────────────────────────────────────
    st.divider()
    info_col, action_col = st.columns(2)

    with info_col:
        st.subheader("📖 Disease Description")
        st.info(context["description"])

        st.subheader("🧬 Cause")
        st.info(context["cause"])

        st.subheader("🔍 Symptoms")
        st.info(context["symptoms"])

        st.subheader("💊 Suggested Medication")
        st.success(context["medication"])

    with action_col:
        st.subheader("🚨 Severity")
        if context["severity"] == "High":
            st.error(context["severity"])
        elif context["severity"] == "Medium":
            st.warning(context["severity"])
        elif context["severity"] == "None":
            st.success(context["severity"])
        else:
            st.info(context["severity"])

        st.subheader("🏥 Treatment Plan")
        st.success(context["treatment"])

        st.subheader("🌿 Sustainability Advice")
        st.info(context["sustainability"])

        st.subheader("🛡️ Prevention Tips")
        st.info(context["prevention"])

    # ── AI Agriculture Advisor ─────────────────────────────────────────────
    st.divider()

    with st.spinner("🤖 Generating AI recommendations..."):
        advisor_data = generate_advisory(
            disease=context["predicted_class"],
            cause=context["cause"],
            symptoms=context["symptoms"],
            medication=context["medication"],
        )

    advisor_source   = advisor_data.get("source",   "local")
    advisor_provider = advisor_data.get("provider", get_current_provider())

    if advisor_source == "gemini":
        st.caption("✨ Advisory powered by **Gemini 2.5 Flash**")
    elif advisor_source == "groq":
        st.caption(f"⚡ Advisory powered by **{advisor_provider}** (Gemini fallback)")
    else:
        st.caption("📚 Advisory powered by **Local Knowledge Base**")

    with st.expander("🤖 AI Agriculture Advisor", expanded=True):
        st.caption(
            "*AI-generated advisory tailored to the detected disease. "
            "Always verify with local agricultural experts.*"
        )

        adv1, adv2 = st.columns(2)

        with adv1:
            st.subheader("🌾 Farmer Explanation")
            st.info(advisor_data["farmer_explanation"])

            st.subheader("🛡️ Preventive Measures")
            st.success(advisor_data["prevention"])

            st.subheader("🌿 Sustainability Advice")
            st.info(advisor_data["sustainability"])

        with adv2:
            st.subheader("💊 Treatment Strategy")
            st.warning(advisor_data["treatment_strategy"])

            st.subheader("📅 Long-Term Recommendations")
            st.info(advisor_data["long_term_advice"])

    # Store for chatbot context
    st.session_state["detected_disease"]         = context["predicted_class"]
    st.session_state["detected_disease_display"] = context["display_name"]

    # ── PDF Download ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("📄 Download Diagnosis Report")
    st.caption("Professionally formatted PDF with all findings and recommendations.")

    report_data = build_report_data(context, image_fingerprint, advisor_data)
    pdf_report  = generate_pdf_report(report_data)

    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_report,
        file_name=f"{report_data['report_id']}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Chatbot section — all native Streamlit components
# ---------------------------------------------------------------------------
def render_chatbot_section():
    st.divider()
    st.subheader("💬 Ask AgriGuard AI")

    current_provider = get_current_provider()
    if current_provider == PROVIDER_GEMINI:
        st.success("🟢 AI Engine: Gemini 2.5 Flash  ·  Powered by Gemini AI + Groq Fallback")
    elif current_provider == PROVIDER_GROQ:
        st.warning("🟡 AI Engine: Groq Llama 3.3 70B  ·  Powered by Gemini AI + Groq Fallback")
    else:
        st.info("📚 AI Engine: Local Advisor  ·  Add API keys in `.env` to enable cloud AI")

    detected_disease = st.session_state.get("detected_disease", "")
    detected_display = st.session_state.get("detected_disease_display", "")

    if detected_display:
        st.caption(f"🔍 Smart context active — I know you detected **{detected_display}**. Ask me anything!")
    else:
        st.caption("Upload a leaf image first to unlock disease-specific advice. General farming questions welcome.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask about disease prevention, treatment, fungicides, or sustainable farming…")

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state["chat_history"].append({"role": "user", "content": user_question})

        with st.spinner("💬 AgriGuard AI is thinking..."):
            reply = chat_with_advisor(
                user_question=user_question,
                detected_disease=detected_disease,
                chat_history=st.session_state["chat_history"],
            )

        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
def render_footer():
    st.divider()
    st.write("**🌱 AgriGuard AI** — AI for Sustainable Agriculture")
    st.caption(
        "Built with: TensorFlow · MobileNetV2 · Streamlit · Gemini AI · Groq · Python  \n"
        "Built for: 1M1B AI for Sustainability · College Major Project · Placement Portfolio"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    inject_css()

    crop_config = get_crop_config(ACTIVE_CROP)
    history_df  = load_history()

    render_sidebar(crop_config)
    render_header(crop_config)

    try:
        trained_model = load_trained_model(crop_config["model_path"])
    except Exception as exc:
        st.error(f"Model could not be loaded from `{crop_config['model_path']}`. Please verify the model file exists.")
        st.caption(str(exc))
        render_analytics_dashboard(history_df)
        return

    image_file, source_label = render_image_source_selector()
    image, image_fingerprint = load_leaf_image(image_file)

    if image is None:
        st.info("🍃 Upload or capture a tomato leaf image above to begin AI diagnosis.")
    else:
        # ── Step 1: Local quality check (blur / brightness / resolution) ──
        quality_ok, quality_msg = validate_image_quality(image)
        if not quality_ok:
            render_validation_error(quality_msg)
            st.stop()

        # ── Step 2: Disease prediction ────────────────────────────────────
        try:
            with st.spinner("🔬 Analysing leaf for disease patterns..."):
                result = predict_disease(image, trained_model, crop_config)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            confidence = result["confidence"]

            # ── Step 3: Confidence-gated Gemini Vision validation ─────────
            # run_validation() returns (passed, source); calls st.stop() on failure
            _, vsource = run_validation(image, image_fingerprint, confidence)
            st.session_state["last_validation_source"] = vsource

            context = get_prediction_context(result, crop_config)
            history_df = log_prediction_once(history_df, image_fingerprint, context)
            render_prediction_results(image, source_label, result, context, crop_config, image_fingerprint)

    st.divider()

    st.subheader("📊 Farm Analytics Dashboard")
    render_analytics_dashboard(history_df)

    render_chatbot_section()
    render_footer()


if __name__ == "__main__":
    main()
