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


st.set_page_config(
    page_title="AgriGuard AI",
    page_icon="🌱",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Mobile-Responsive CSS
# ---------------------------------------------------------------------------
def inject_responsive_css():
    """Inject comprehensive responsive CSS for desktop and mobile."""
    st.markdown(
        """
        <style>
        /* ================================================================
           FONTS & ROOT
        ================================================================ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* ================================================================
           GLOBAL CARD STYLE
        ================================================================ */
        .agri-card {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }

        /* ================================================================
           HEADER MOBILE STACK
        ================================================================ */
        .agri-header {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.75rem;
            padding: 1rem 0;
        }

        .agri-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }

        .agri-badge-green {
            background: rgba(34, 197, 94, 0.18);
            color: #4ade80;
            border: 1px solid rgba(34, 197, 94, 0.35);
        }

        .agri-badge-red {
            background: rgba(239, 68, 68, 0.18);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.35);
        }

        .agri-badge-blue {
            background: rgba(59, 130, 246, 0.18);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.35);
        }

        /* ================================================================
           METRIC CARDS — 4 cols desktop → 2 cols tablet → 1 col mobile
        ================================================================ */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 1rem;
            margin: 1rem 0;
        }

        .metric-card {
            background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 14px;
            padding: 1rem 1.25rem;
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            cursor: default;
        }

        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 28px rgba(0,0,0,0.4);
        }

        .metric-card .metric-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: rgba(255,255,255,0.5);
            margin-bottom: 0.4rem;
        }

        .metric-card .metric-value {
            font-size: 1.15rem;
            font-weight: 700;
            color: #e2e8f0;
        }

        /* ================================================================
           CHAT UI — mobile-friendly
        ================================================================ */
        .chat-container {
            max-width: 100%;
            overflow-x: hidden;
        }

        /* Ensure chat input stays accessible */
        section[data-testid="stChatInput"] {
            position: sticky;
            bottom: 0;
            background: var(--background-color, #0e1117);
            padding-bottom: 0.5rem;
            z-index: 100;
        }

        /* Chat message bubbles */
        [data-testid="stChatMessage"] {
            max-width: 100%;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }

        [data-testid="stChatMessage"] p {
            white-space: pre-wrap;
            word-break: break-word;
        }

        /* ================================================================
           DATAFRAMES — horizontal scroll on small screens
        ================================================================ */
        [data-testid="stDataFrame"] {
            overflow-x: auto;
            max-width: 100%;
        }

        /* ================================================================
           DOWNLOAD BUTTON — full-width on mobile
        ================================================================ */
        .download-btn-wrap [data-testid="stDownloadButton"] button {
            width: 100%;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-size: 1rem;
            font-weight: 600;
            background: linear-gradient(135deg, #22c55e, #16a34a);
            color: white;
            border: none;
            transition: transform 0.15s, box-shadow 0.15s;
        }

        .download-btn-wrap [data-testid="stDownloadButton"] button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(34, 197, 94, 0.4);
        }

        /* ================================================================
           PLOTLY CHARTS — responsive
        ================================================================ */
        .js-plotly-plot, .plotly {
            width: 100% !important;
            min-width: 0 !important;
        }

        /* ================================================================
           RESPONSIVE BREAKPOINTS
        ================================================================ */

        /* Tablet: ≤ 900px */
        @media (max-width: 900px) {
            .metric-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }

        /* Mobile: ≤ 768px */
        @media (max-width: 768px) {
            /* Font size adjustments */
            h1 { font-size: 1.6rem !important; }
            h2 { font-size: 1.25rem !important; }
            h3 { font-size: 1.05rem !important; }

            /* Full-width metric cards */
            .metric-grid {
                grid-template-columns: 1fr;
            }

            /* Reduce padding on cards */
            .agri-card {
                padding: 0.9rem 1rem;
            }

            /* Ensure columns stack on mobile */
            [data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
            }

            [data-testid="stHorizontalBlock"] > div {
                min-width: min(100%, 280px);
            }

            /* Shrink table text */
            [data-testid="stDataFrame"] th,
            [data-testid="stDataFrame"] td {
                font-size: 0.78rem !important;
                padding: 0.35rem 0.5rem !important;
            }

            /* Chat adjustments */
            [data-testid="stChatMessage"] {
                margin-left: 0 !important;
                margin-right: 0 !important;
            }

            /* Download button full-width */
            .download-btn-wrap {
                display: block;
                width: 100%;
            }

            .download-btn-wrap [data-testid="stDownloadButton"] {
                width: 100%;
            }

            /* Reduce main padding */
            .main .block-container {
                padding-left: 0.75rem !important;
                padding-right: 0.75rem !important;
                padding-top: 1rem !important;
            }

            /* Stack image + prediction vertically */
            section[data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] {
                flex-direction: column;
            }
        }

        /* Very small phones: ≤ 480px */
        @media (max-width: 480px) {
            h1 { font-size: 1.35rem !important; }

            .agri-badge {
                font-size: 0.72rem;
                padding: 0.28rem 0.65rem;
            }
        }

        /* ================================================================
           MISC POLISH
        ================================================================ */
        /* Smooth all transitions */
        * {
            transition-timing-function: cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Progress bar polish */
        [data-testid="stProgress"] > div > div {
            border-radius: 999px;
            background: linear-gradient(90deg, #22c55e, #4ade80) !important;
        }

        /* Expander header */
        [data-testid="stExpander"] summary {
            font-weight: 600;
            font-size: 1rem;
        }

        /* Status banner margins */
        [data-testid="stAlert"] {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_trained_model(model_path):
    """Load the TensorFlow model once per Streamlit session."""
    return tf.keras.models.load_model(model_path)


def display_name_for(class_name, crop_config):
    """Return a readable disease name for UI and reports."""
    return crop_config["display_names"].get(
        class_name,
        class_name.replace("_", " "),
    )


def load_leaf_image(image_file):
    """Read either an uploaded file or camera capture into a PIL image."""
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
    """Run the shared prediction workflow for upload and camera inputs."""
    class_names = crop_config["class_names"]
    image_size = crop_config.get("image_size", (224, 224))

    resized_image = image.resize(image_size)
    image_array = np.array(resized_image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    prediction = trained_model.predict(image_array, verbose=0)
    probabilities = prediction[0].astype(float)

    if len(probabilities) != len(class_names):
        raise ValueError(
            "Model output size does not match the configured disease classes."
        )

    top_indices = np.argsort(probabilities)[::-1][:5]
    predicted_class = class_names[int(top_indices[0])]
    confidence = float(probabilities[int(top_indices[0])] * 100)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "top_indices": top_indices,
    }


def build_probability_table(probabilities, crop_config):
    """Create the full class probability table."""
    class_names = crop_config["class_names"]

    return pd.DataFrame(
        {
            "Disease": [
                display_name_for(class_name, crop_config)
                for class_name in class_names
            ],
            "Probability (%)": [
                round(float(probability) * 100, 2)
                for probability in probabilities
            ],
        }
    )


def build_top_prediction_table(result, crop_config):
    """Create the top-5 prediction table used for the comparison chart."""
    class_names = crop_config["class_names"]

    prediction_data = []
    for index in result["top_indices"]:
        class_name = class_names[int(index)]
        prediction_data.append(
            {
                "Disease": display_name_for(class_name, crop_config),
                "Confidence": round(
                    float(result["probabilities"][int(index)] * 100),
                    2,
                ),
            }
        )

    return pd.DataFrame(prediction_data)


def render_provider_sidebar():
    """Render the AI provider status panel in the Streamlit sidebar."""
    status = get_provider_status()
    current = get_current_provider()

    with st.sidebar:
        st.markdown("### 🤖 AI Provider Status")

        # Current provider highlight
        if current == PROVIDER_GEMINI:
            st.success(f"**Active:** {PROVIDER_GEMINI}")
        elif current == PROVIDER_GROQ:
            st.warning(f"**Active:** {PROVIDER_GROQ}")
        else:
            st.error(f"**Active:** {PROVIDER_LOCAL}")

        st.markdown("---")
        st.markdown("**Provider availability:**")

        # Gemini row
        if status["gemini_available"]:
            st.markdown("🟢 Gemini 2.5 Flash — **ready**")
        else:
            st.markdown(
                f"🔴 Gemini 2.5 Flash — unavailable  \n"
                f"<small>{status['gemini_error']}</small>",
                unsafe_allow_html=True,
            )

        # Groq row
        if status["groq_available"]:
            st.markdown("🟢 Groq Llama 3.3 70B — **ready**")
        else:
            st.markdown(
                f"🔴 Groq Llama 3.3 70B — unavailable  \n"
                f"<small>{status['groq_error']}</small>",
                unsafe_allow_html=True,
            )

        # Local KB — always available
        st.markdown("🟢 Local Knowledge Base — **always ready**")

        st.markdown("---")
        st.caption(
            "Priority order: Gemini → Groq → Local KB. "
            "Keys are loaded from `.env`."
        )


def render_gemini_status_badge():
    """Render a small coloured badge in the header showing active provider."""
    provider = get_current_provider()
    if provider == PROVIDER_GEMINI:
        st.markdown(
            '<span class="agri-badge agri-badge-green">🟢 Gemini Connected</span>',
            unsafe_allow_html=True,
        )
    elif provider == PROVIDER_GROQ:
        st.markdown(
            '<span class="agri-badge agri-badge-blue">🟡 Groq Active (Gemini fallback)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="agri-badge agri-badge-red">🔴 Local KB · No cloud provider</span>',
            unsafe_allow_html=True,
        )


def render_header(crop_config):
    """Render the application header and model summary."""
    st.title("🌱 AgriGuard AI")
    st.caption("AI-Powered Sustainable Agriculture Assistant")

    # ── Gemini status badge (always visible) ──────────────────────────────
    render_gemini_status_badge()

    st.markdown(
        """
### AI-Powered Sustainable Agriculture Assistant

Upload or capture a tomato leaf image to:
- Detect diseases
- Get treatment recommendations
- Receive sustainability insights
- Support smart farming decisions
"""
    )

    # ── Responsive metric grid ─────────────────────────────────────────────
    active_provider = get_current_provider()
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Disease Classes</div>
                <div class="metric-value">{len(crop_config["class_names"])}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Model</div>
                <div class="metric-value">{crop_config["model_name"]}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Crop</div>
                <div class="metric-value">{crop_config["crop_name"]}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">AI Advisor</div>
                <div class="metric-value">{active_provider}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Provider status banner ─────────────────────────────────────────────
    status_text = get_gemini_status()
    if active_provider == PROVIDER_GEMINI:
        st.success(f"**{status_text}** — AI Advisor powered by Gemini 2.5 Flash")
    elif active_provider == PROVIDER_GROQ:
        st.warning(
            f"**{status_text}** — Gemini quota exceeded or unavailable; "
            "using Groq Llama 3.3 70B as fallback."
        )
    else:
        st.warning(
            f"**{status_text}**\n\n"
            "To enable cloud AI: open `.env` and add `GEMINI_API_KEY` or `GROQ_API_KEY`. "
            "Get Gemini key at [aistudio.google.com](https://aistudio.google.com/app/apikey), "
            "Groq key at [console.groq.com](https://console.groq.com/)."
        )


def render_image_source_selector():
    """Render upload and camera options while returning one selected input."""
    st.write("## Choose Image Source")

    with st.container():
        source = st.radio(
            "Choose Image Source",
            ["Upload Image", "Capture from Camera"],
            horizontal=True,
            label_visibility="collapsed",
        )

        input_col, note_col = st.columns([1.4, 1])

        with input_col:
            if source == "Upload Image":
                image_file = st.file_uploader(
                    "Upload Tomato Leaf Image",
                    type=["jpg", "jpeg", "png"],
                )
            else:
                image_file = st.camera_input("Capture Tomato Leaf Image")

        with note_col:
            st.info("Use a clear tomato leaf image for a more reliable result.")

    return image_file, source


def get_prediction_context(result, crop_config):
    """Collect disease knowledge, recommendations, and scoring for the result."""
    predicted_class = result["predicted_class"]
    display_name = display_name_for(predicted_class, crop_config)
    sustainability_score = crop_config["sustainability_scores"].get(
        predicted_class,
        60,
    )
    knowledge = crop_knowledge.get(predicted_class, {})
    disease_data = recommendations.get(predicted_class, {})

    return {
        "predicted_class": predicted_class,
        "display_name": display_name,
        "confidence": result["confidence"],
        "sustainability_score": sustainability_score,
        "description": disease_info.get(predicted_class, "Information unavailable."),
        "cause": knowledge.get("cause", "Information unavailable."),
        "symptoms": knowledge.get("symptoms", "Information unavailable."),
        "medication": knowledge.get("medication", "Information unavailable."),
        "severity": disease_data.get("severity", "Unknown"),
        "treatment": disease_data.get("treatment", "Information unavailable."),
        "sustainability": disease_data.get(
            "sustainability",
            "Information unavailable.",
        ),
        "prevention": disease_data.get("prevention", "Information unavailable."),
    }


def log_prediction_once(history_df, image_fingerprint, context):
    """Append new predictions while avoiding repeated logs during reruns."""
    if "logged_prediction_keys" not in st.session_state:
        st.session_state["logged_prediction_keys"] = set()

    log_key = (
        f"{image_fingerprint}:"
        f"{context['predicted_class']}:"
        f"{context['confidence']:.2f}"
    )

    if log_key in st.session_state["logged_prediction_keys"]:
        return history_df

    try:
        updated_history = append_prediction(
            disease_name=context["display_name"],
            confidence=context["confidence"],
            sustainability_score=context["sustainability_score"],
        )
        st.session_state["logged_prediction_keys"].add(log_key)
        return updated_history
    except Exception as exc:
        st.warning(f"Prediction history could not be updated: {exc}")
        return history_df


def render_severity(severity):
    """Render severity with simple visual status feedback."""
    if severity == "High":
        st.error(severity)
    elif severity == "Medium":
        st.warning(severity)
    elif severity == "None":
        st.success(severity)
    else:
        st.info(severity)


def build_report_data(context, image_fingerprint, advisor_data=None):
    """Prepare report data for the professional PDF generator."""
    generated_at_value = datetime.now()
    generated_at = generated_at_value.strftime("%Y-%m-%d %H:%M:%S")
    report_time = generated_at_value.strftime("%Y%m%d%H%M%S")
    report_id = f"AGR-{report_time}-{image_fingerprint[:6].upper()}"

    report = {
        "report_id": report_id,
        "generated_at": generated_at,
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


def render_prediction_results(
    image,
    source_label,
    result,
    context,
    crop_config,
    image_fingerprint,
):
    """Render diagnosis result, recommendations, charts, and PDF download."""

    # ── Image + result side-by-side (stacks on mobile via CSS) ────────────
    result_col, image_col = st.columns([1.05, 1])

    with image_col:
        st.image(
            image,
            caption=f"{source_label} Leaf Image",
            use_container_width=True,
        )

    with result_col:
        st.success(f"Disease Detected: {context['display_name']}")

        confidence_progress = min(max(context["confidence"] / 100, 0), 1)
        sustainability_progress = min(
            max(context["sustainability_score"] / 100, 0),
            1,
        )

        st.write("## Confidence Score")
        st.progress(confidence_progress)
        st.info(f"{context['confidence']:.2f}%")

        st.write("## Sustainability Score")
        st.progress(sustainability_progress)
        st.write(f"{context['sustainability_score']}/100")

    # ── Probability tables + chart ─────────────────────────────────────────
    probability_table = build_probability_table(
        result["probabilities"],
        crop_config,
    )
    top_prediction_table = build_top_prediction_table(result, crop_config)

    score_col, chart_col = st.columns(2)

    with score_col:
        st.write("## Detailed Prediction Scores")
        st.dataframe(
            probability_table.sort_values(
                by="Probability (%)",
                ascending=False,
            ),
            use_container_width=True,
        )

    with chart_col:
        st.write("## Top 5 Predictions")
        st.dataframe(top_prediction_table, use_container_width=True)

        st.write("## Confidence Comparison")
        st.bar_chart(top_prediction_table.set_index("Disease"))

    # ── Disease info + treatment columns ──────────────────────────────────
    info_col, action_col = st.columns(2)

    with info_col:
        st.write("## Disease Description")
        st.info(context["description"])

        st.write("## Cause")
        st.info(context["cause"])

        st.write("## Symptoms")
        st.info(context["symptoms"])

        st.write("## Suggested Medication")
        st.success(context["medication"])

    with action_col:
        st.write("## Severity")
        render_severity(context["severity"])

        st.write("## Treatment")
        st.success(context["treatment"])

        st.write("## Sustainability Advice")
        st.info(context["sustainability"])

        st.write("## Prevention Tips")
        st.info(context["prevention"])

    # ── AI Agriculture Advisor Section ─────────────────────────────────────
    with st.spinner("🤖 Generating AI recommendations..."):
        advisor_data = generate_advisory(
            disease=context["predicted_class"],
            cause=context["cause"],
            symptoms=context["symptoms"],
            medication=context["medication"],
        )

    advisor_source = advisor_data.get("source", "local")
    advisor_provider = advisor_data.get("provider", get_current_provider())
    if advisor_source == "gemini":
        st.caption("✨ Advisory powered by **Gemini 2.5 Flash**")
    elif advisor_source == "groq":
        st.caption(f"⚡ Advisory powered by **{advisor_provider}** (Gemini fallback)")
    else:
        st.caption(
            "📚 Advisory powered by **Local Knowledge Base** "
            "(add GEMINI_API_KEY or GROQ_API_KEY in `.env` to enable cloud AI)"
        )

    with st.expander("🤖 AI Agriculture Advisor", expanded=True):
        st.markdown(
            "> *AI-generated advisory tailored to the detected disease. "
            "Always verify recommendations with local experts.*"
        )

        # On mobile, single column; on desktop, two columns
        adv_col1, adv_col2 = st.columns(2)

        with adv_col1:
            st.markdown("### 🌾 Farmer Explanation")
            st.info(advisor_data["farmer_explanation"])

            st.markdown("### 🛡️ Prevention Tips")
            st.success(advisor_data["prevention"])

            st.markdown("### 🌿 Sustainability Advice")
            st.info(advisor_data["sustainability"])

        with adv_col2:
            st.markdown("### 💊 Treatment Strategy")
            st.warning(advisor_data["treatment_strategy"])

            st.markdown("### 📅 Long-Term Advice")
            st.info(advisor_data["long_term_advice"])

    # Store detected disease in session state for chatbot smart context.
    st.session_state["detected_disease"] = context["predicted_class"]
    st.session_state["detected_disease_display"] = context["display_name"]

    # ── PDF Download — full-width highlighted button ───────────────────────
    report_data = build_report_data(context, image_fingerprint, advisor_data)
    pdf_report = generate_pdf_report(report_data)

    st.markdown("---")
    st.markdown("### 📄 Download Your Report")
    st.markdown('<div class="download-btn-wrap">', unsafe_allow_html=True)
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_report,
        file_name=f"{report_data['report_id']}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    inject_responsive_css()

    crop_config = get_crop_config(ACTIVE_CROP)
    history_df = load_history()

    # Sidebar: AI provider status panel
    render_provider_sidebar()

    render_header(crop_config)

    # Model loading is guarded so missing model files produce a friendly message.
    try:
        trained_model = load_trained_model(crop_config["model_path"])
    except Exception as exc:
        st.error(
            f"Model could not be loaded from {crop_config['model_path']}. "
            "Please verify the model file exists."
        )
        st.caption(str(exc))
        render_analytics_dashboard(history_df)
        return

    # Image source section supports upload and camera capture with one workflow.
    image_file, source_label = render_image_source_selector()
    image, image_fingerprint = load_leaf_image(image_file)

    if image is None:
        st.info("Upload or capture a tomato leaf image to begin diagnosis.")
    else:
        try:
            with st.spinner("🔍 Analyzing leaf image..."):
                result = predict_disease(image, trained_model, crop_config)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            context = get_prediction_context(result, crop_config)

            # Prediction history is appended once per unique image in this session.
            history_df = log_prediction_once(
                history_df,
                image_fingerprint,
                context,
            )

            render_prediction_results(
                image,
                source_label,
                result,
                context,
                crop_config,
                image_fingerprint,
            )

    st.divider()

    # Farm analytics dashboard.
    render_analytics_dashboard(history_df)

    # ── Ask AgriGuard AI — Chatbot Section ─────────────────────────────────
    st.divider()
    st.markdown("## 💬 Ask AgriGuard AI")

    detected_disease = st.session_state.get("detected_disease", "")
    detected_display = st.session_state.get("detected_disease_display", "")

    if detected_display:
        st.caption(
            f"🔍 Smart context active — I know you detected **{detected_display}**. "
            "Ask me anything about it!"
        )
    else:
        st.caption(
            "Upload or capture a leaf image first to unlock disease-specific advice. "
            "You can still ask general farming questions below."
        )

    # Chatbot backend badge
    st.caption(f"Powered by: **{get_current_provider()}**")

    # Initialise chat history.
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Render existing chat messages inside a scroll-safe container.
    with st.container():
        for message in st.session_state["chat_history"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Accept new user input.
    user_question = st.chat_input(
        "Ask about disease prevention, treatment, fungicides, or sustainable farming…"
    )

    if user_question:
        # Display user message.
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state["chat_history"].append(
            {"role": "user", "content": user_question}
        )

        # Generate advisor response.
        with st.spinner("💬 AgriGuard AI is thinking..."):
            advisor_reply = chat_with_advisor(
                user_question=user_question,
                detected_disease=detected_disease,
                chat_history=st.session_state["chat_history"],
            )

        # Display assistant message.
        with st.chat_message("assistant"):
            st.markdown(advisor_reply)
        st.session_state["chat_history"].append(
            {"role": "assistant", "content": advisor_reply}
        )


if __name__ == "__main__":
    main()
