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
import html as _html
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
    layout="wide",
)


# ---------------------------------------------------------------------------
# HTML-escape helper (safe injection of dynamic text into HTML blocks)
# ---------------------------------------------------------------------------
def _e(text: str) -> str:
    """Escape a value for safe embedding inside an HTML attribute or body."""
    return _html.escape(str(text))


# ---------------------------------------------------------------------------
# Design-System CSS
# ---------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* ── No horizontal scroll ─────────────────────────────────────── */
        html, body {
            overflow-x: hidden !important;
            max-width: 100vw !important;
        }
        .main .block-container {
            max-width: 100% !important;
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
            padding-bottom: 4rem !important;
            box-sizing: border-box !important;
        }

        /* ── Glassmorphism card ───────────────────────────────────────── */
        .ag-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(76, 175, 80, 0.14);
            border-radius: 20px;
            padding: 28px 24px;
            margin: 0 0 24px 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.40),
                        inset 0 1px 0 rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            width: 100%;
            box-sizing: border-box;
        }

        /* ── Section header ───────────────────────────────────────────── */
        .section-header {
            display: flex;
            align-items: flex-start;
            gap: 12px;
            margin-bottom: 20px;
            padding-bottom: 16px;
            border-bottom: 1px solid rgba(76, 175, 80, 0.11);
        }
        .section-header-text { flex: 1; }
        .section-title {
            font-size: clamp(1.2rem, 4vw, 1.6rem);
            font-weight: 800;
            color: #ffffff;
            letter-spacing: -0.02em;
            margin: 0 0 4px 0;
            line-height: 1.2;
        }
        .section-desc {
            font-size: 0.87rem;
            color: rgba(255,255,255,0.48);
            margin: 0;
        }
        .section-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background: rgba(76, 175, 80, 0.11);
            border: 1px solid rgba(76, 175, 80, 0.24);
            border-radius: 999px;
            padding: 5px 13px;
            font-size: 0.74rem;
            font-weight: 700;
            color: #81C784;
            white-space: nowrap;
            flex-shrink: 0;
        }

        /* ── Hero banner ──────────────────────────────────────────────── */
        .hero-banner {
            background: linear-gradient(135deg,
                #1B4332 0%, #2E7D32 30%, #0d1f0d 65%, #0A0F0A 100%);
            border: 1px solid rgba(76, 175, 80, 0.20);
            border-radius: 24px;
            padding: 52px 44px 44px;
            margin: 0 0 28px 0;
            position: relative;
            overflow: hidden;
            box-shadow: 0 24px 64px rgba(0,0,0,0.55),
                        inset 0 1px 0 rgba(255,255,255,0.08);
        }
        .hero-banner::before {
            content: '';
            position: absolute;
            top: -40%; right: -5%;
            width: 480px; height: 480px;
            background: radial-gradient(circle, rgba(76,175,80,0.10) 0%, transparent 65%);
            pointer-events: none;
        }
        .hero-banner::after {
            content: '';
            position: absolute;
            bottom: -25%; left: 25%;
            width: 320px; height: 320px;
            background: radial-gradient(circle, rgba(212,160,23,0.06) 0%, transparent 65%);
            pointer-events: none;
        }
        .hero-eyebrow {
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            color: #81C784;
            text-transform: uppercase;
            margin: 0 0 12px 0;
        }
        .hero-logo {
            font-size: clamp(2.3rem, 7vw, 3.8rem);
            font-weight: 900;
            color: #ffffff;
            letter-spacing: -0.04em;
            line-height: 1;
            margin: 0 0 14px 0;
        }
        .hero-tagline {
            font-size: clamp(0.98rem, 3vw, 1.28rem);
            font-weight: 400;
            color: rgba(255,255,255,0.66);
            margin: 0 0 30px 0;
            line-height: 1.55;
            max-width: 560px;
        }
        .hero-powered {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-bottom: 36px;
        }
        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 999px;
            padding: 8px 16px;
            font-size: 0.82rem;
            font-weight: 600;
            color: rgba(255,255,255,0.84);
            backdrop-filter: blur(4px);
        }
        .hero-dot {
            width: 7px; height: 7px;
            border-radius: 50%;
            background: #4CAF50;
            flex-shrink: 0;
        }
        .hero-stats {
            display: flex;
            flex-wrap: wrap;
            gap: 14px;
        }
        .hero-stat {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 14px;
            padding: 14px 18px;
            min-width: 90px;
            text-align: center;
        }
        .hero-stat-val {
            font-size: clamp(1.1rem, 3vw, 1.5rem);
            font-weight: 800;
            color: #4CAF50;
            line-height: 1;
            display: block;
            word-break: break-word;
        }
        .hero-stat-lbl {
            font-size: 0.69rem;
            color: rgba(255,255,255,0.46);
            font-weight: 500;
            margin-top: 5px;
            display: block;
            text-transform: uppercase;
            letter-spacing: 0.07em;
        }
        .provider-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border-radius: 999px;
            padding: 8px 18px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-top: 24px;
        }
        .provider-pill.gemini {
            background: rgba(76,175,80,0.13);
            border: 1px solid rgba(76,175,80,0.30);
            color: #81C784;
        }
        .provider-pill.groq {
            background: rgba(255,152,0,0.11);
            border: 1px solid rgba(255,152,0,0.26);
            color: #FFB74D;
        }
        .provider-pill.local {
            background: rgba(33,150,243,0.11);
            border: 1px solid rgba(33,150,243,0.26);
            color: #64B5F6;
        }

        /* ── Camera card ──────────────────────────────────────────────── */
        .camera-card {
            background: linear-gradient(145deg, #0a1a0a 0%, #142414 45%, #0c1c2c 100%);
            border: 1.5px solid rgba(76, 175, 80, 0.22);
            border-radius: 18px;
            padding: 24px 20px 20px;
            margin: 0 0 16px 0;
            box-shadow: 0 10px 36px rgba(0,0,0,0.5),
                        inset 0 1px 0 rgba(255,255,255,0.05);
            width: 100%;
            box-sizing: border-box;
        }
        .camera-card-heading {
            font-size: clamp(1.35rem, 5vw, 1.85rem);
            font-weight: 800;
            color: #ffffff;
            letter-spacing: -0.02em;
            margin: 0 0 6px 0;
            line-height: 1.2;
        }
        .camera-card-subtitle {
            font-size: 0.87rem;
            color: rgba(255,255,255,0.46);
            margin: 0 0 18px 0;
        }
        .camera-instructions {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 20px;
        }
        .camera-tip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(76, 175, 80, 0.08);
            border: 1px solid rgba(76, 175, 80, 0.18);
            border-radius: 999px;
            padding: 6px 13px;
            font-size: 0.81rem;
            color: rgba(255,255,255,0.80);
            white-space: nowrap;
            backdrop-filter: blur(4px);
        }
        .camera-tip-icon { font-size: 0.93rem; line-height: 1; }

        /* ── Camera widget — full width ───────────────────────────────── */
        [data-testid="stCameraInput"] {
            width: 100% !important;
            max-width: 100% !important;
            display: block;
        }
        [data-testid="stCameraInput"] > div {
            width: 100% !important;
            max-width: 100% !important;
        }
        [data-testid="stCameraInput"] video,
        [data-testid="stCameraInput"] canvas,
        [data-testid="stCameraInput"] img {
            width: 100% !important;
            max-width: 100% !important;
            border-radius: 14px;
            display: block;
        }
        [data-testid="stCameraInput"] > label { display: none !important; }
        [data-testid="stCameraInput"] > div > div {
            aspect-ratio: 4 / 3;
            border-radius: 14px;
            overflow: hidden;
        }

        /* ── File uploader ────────────────────────────────────────────── */
        [data-testid="stFileUploader"] {
            width: 100% !important;
            max-width: 100% !important;
        }

        /* ── Diagnosis metric grid ────────────────────────────────────── */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 14px;
            margin: 20px 0 24px 0;
        }
        .metric-card {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 16px;
            padding: 18px 14px 16px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            border-radius: 16px 16px 0 0;
        }
        .metric-card.healthy::before { background: linear-gradient(90deg,#4CAF50,#81C784); }
        .metric-card.low::before     { background: linear-gradient(90deg,#8BC34A,#C5E1A5); }
        .metric-card.medium::before  { background: linear-gradient(90deg,#FF9800,#FFB74D); }
        .metric-card.high::before    { background: linear-gradient(90deg,#F44336,#EF9A9A); }
        .metric-card.info::before    { background: linear-gradient(90deg,#4CAF50,#D4A017); }
        .metric-card.sust::before    { background: linear-gradient(90deg,#2E7D32,#4CAF50); }
        .mc-icon { font-size: 1.5rem; margin-bottom: 8px; display: block; }
        .mc-val {
            font-size: 1.18rem;
            font-weight: 800;
            color: #ffffff;
            line-height: 1.15;
            display: block;
            margin-bottom: 5px;
            word-break: break-word;
        }
        .mc-lbl {
            font-size: 0.68rem;
            color: rgba(255,255,255,0.46);
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            display: block;
        }

        /* ── Severity badges ──────────────────────────────────────────── */
        .sev-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border-radius: 999px;
            padding: 5px 14px;
            font-size: 0.81rem;
            font-weight: 700;
            letter-spacing: 0.03em;
        }
        .sev-badge.healthy { background:rgba(76,175,80,0.14);  color:#81C784; border:1px solid rgba(76,175,80,0.28);  }
        .sev-badge.low     { background:rgba(139,195,74,0.14); color:#AED581; border:1px solid rgba(139,195,74,0.28); }
        .sev-badge.medium  { background:rgba(255,152,0,0.14);  color:#FFB74D; border:1px solid rgba(255,152,0,0.28);  }
        .sev-badge.high    { background:rgba(244,67,54,0.14);  color:#EF9A9A; border:1px solid rgba(244,67,54,0.28);  }
        .sev-badge.info    { background:rgba(76,175,80,0.10);  color:#A5D6A7; border:1px solid rgba(76,175,80,0.20);  }

        /* ── Validation source badge ──────────────────────────────────── */
        .vsource {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border-radius: 999px;
            padding: 5px 13px;
            font-size: 0.79rem;
            font-weight: 600;
            margin-top: 10px;
        }
        .vsource.local  { background:rgba(76,175,80,0.11);  color:#81C784; border:1px solid rgba(76,175,80,0.24);  }
        .vsource.gemini { background:rgba(33,150,243,0.11); color:#64B5F6; border:1px solid rgba(33,150,243,0.24); }
        .vsource.fb     { background:rgba(255,152,0,0.11);  color:#FFB74D; border:1px solid rgba(255,152,0,0.24);  }

        /* ── Disease detail cards ─────────────────────────────────────── */
        .detail-card {
            background: rgba(255,255,255,0.025);
            border: 1px solid rgba(76,175,80,0.09);
            border-radius: 14px;
            padding: 18px 16px;
            margin-bottom: 12px;
        }
        .detail-card-title {
            font-size: 0.75rem;
            font-weight: 700;
            color: #81C784;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            margin: 0 0 8px 0;
        }
        .detail-card-body {
            font-size: 0.87rem;
            color: rgba(255,255,255,0.76);
            line-height: 1.65;
        }

        /* ── Advisor grid ─────────────────────────────────────────────── */
        .advisor-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 16px;
            margin-top: 4px;
        }
        .advisor-card {
            background: rgba(255,255,255,0.028);
            border: 1px solid rgba(76,175,80,0.11);
            border-radius: 16px;
            padding: 20px 18px;
            position: relative;
            overflow: hidden;
        }
        .advisor-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, #4CAF50, #D4A017);
        }
        .advisor-card-title {
            font-size: 0.87rem;
            font-weight: 700;
            color: #81C784;
            margin: 0 0 10px 0;
        }
        .advisor-card-body {
            font-size: 0.85rem;
            color: rgba(255,255,255,0.74);
            line-height: 1.62;
        }

        /* ── Reports card ─────────────────────────────────────────────── */
        .report-card {
            background: linear-gradient(135deg,
                rgba(46,125,50,0.13) 0%, rgba(212,160,23,0.07) 100%);
            border: 1px solid rgba(76,175,80,0.20);
            border-radius: 16px;
            padding: 26px 22px;
            text-align: center;
        }
        .report-card-title {
            font-size: 1.18rem;
            font-weight: 800;
            color: #ffffff;
            margin: 0 0 8px 0;
        }
        .report-card-desc {
            font-size: 0.86rem;
            color: rgba(255,255,255,0.50);
            margin: 0 0 22px 0;
            line-height: 1.5;
        }

        /* ── Chatbot provider / context badges ────────────────────────── */
        .chat-provider {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            border-radius: 999px;
            padding: 7px 16px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .chat-provider.gemini { background:rgba(76,175,80,0.13);  border:1px solid rgba(76,175,80,0.28);  color:#81C784; }
        .chat-provider.groq   { background:rgba(255,152,0,0.11);  border:1px solid rgba(255,152,0,0.26);  color:#FFB74D; }
        .chat-provider.local  { background:rgba(33,150,243,0.11); border:1px solid rgba(33,150,243,0.26); color:#64B5F6; }
        .chat-context {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            background: rgba(212,160,23,0.10);
            border: 1px solid rgba(212,160,23,0.22);
            border-radius: 999px;
            padding: 5px 13px;
            font-size: 0.79rem;
            font-weight: 600;
            color: #F6CC5D;
            margin-left: 8px;
        }

        /* ── Progress bars ────────────────────────────────────────────── */
        [data-testid="stProgress"] > div > div {
            border-radius: 999px;
            background: linear-gradient(90deg, #2E7D32, #4CAF50) !important;
        }

        /* ── Expanders ────────────────────────────────────────────────── */
        [data-testid="stExpander"] summary { font-weight: 600; }
        [data-testid="stExpander"] {
            background: rgba(255,255,255,0.02) !important;
            border: 1px solid rgba(76,175,80,0.09) !important;
            border-radius: 12px !important;
        }

        /* ── Buttons ──────────────────────────────────────────────────── */
        [data-testid="stButton"] > button {
            min-height: 52px !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            border-radius: 12px !important;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
            width: 100%;
        }
        [data-testid="stButton"] > button:active { transform: scale(0.97); }

        /* ── Download button ──────────────────────────────────────────── */
        [data-testid="stDownloadButton"] > button {
            min-height: 56px !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            border-radius: 14px !important;
            background: linear-gradient(135deg, #2E7D32, #4CAF50) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 4px 20px rgba(76,175,80,0.28) !important;
            transition: box-shadow 0.2s ease, transform 0.15s ease !important;
            width: 100%;
        }
        [data-testid="stDownloadButton"] > button:hover {
            box-shadow: 0 8px 30px rgba(76,175,80,0.46) !important;
            transform: translateY(-1px);
        }

        /* ── Sidebar ──────────────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0A1A0A 0%, #080E08 100%) !important;
            border-right: 1px solid rgba(76,175,80,0.09) !important;
        }
        .sb-logo {
            font-size: 1.28rem;
            font-weight: 900;
            color: #4CAF50;
            letter-spacing: -0.02em;
        }
        .sb-version {
            font-size: 0.72rem;
            color: rgba(255,255,255,0.35);
            font-weight: 500;
            margin-top: 2px;
        }
        .sb-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border-radius: 999px;
            padding: 5px 12px;
            font-size: 0.77rem;
            font-weight: 600;
            margin: 3px 0;
        }
        .sb-pill.g { background:rgba(76,175,80,0.13);  color:#81C784; border:1px solid rgba(76,175,80,0.26); }
        .sb-pill.q { background:rgba(255,152,0,0.11);  color:#FFB74D; border:1px solid rgba(255,152,0,0.24); }
        .sb-pill.l { background:rgba(33,150,243,0.11); color:#64B5F6; border:1px solid rgba(33,150,243,0.24);}
        .sb-chip {
            display: inline-block;
            background: rgba(76,175,80,0.07);
            border: 1px solid rgba(76,175,80,0.14);
            border-radius: 999px;
            padding: 4px 11px;
            font-size: 0.74rem;
            color: rgba(255,255,255,0.62);
            margin: 2px 2px;
        }

        /* ── st.metric styling ────────────────────────────────────────── */
        [data-testid="stMetric"] {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 12px;
            padding: 12px 16px;
        }
        [data-testid="stMetricValue"] { color: #4CAF50 !important; }

        /* ── Chat messages ────────────────────────────────────────────── */
        [data-testid="stChatMessage"] {
            background: rgba(255,255,255,0.025) !important;
            border: 1px solid rgba(76,175,80,0.08) !important;
            border-radius: 14px !important;
            margin: 6px 0 !important;
        }

        /* ── Divider ──────────────────────────────────────────────────── */
        hr { border-color: rgba(76,175,80,0.09) !important; }

        /* ── Footer ───────────────────────────────────────────────────── */
        .ag-footer {
            background: rgba(10,26,10,0.72);
            border: 1px solid rgba(76,175,80,0.09);
            border-radius: 18px;
            padding: 28px 24px;
            margin-top: 36px;
            text-align: center;
        }
        .ag-footer-logo {
            font-size: 1.18rem;
            font-weight: 800;
            color: #4CAF50;
            letter-spacing: -0.02em;
            margin-bottom: 6px;
        }
        .ag-footer-tagline {
            font-size: 0.79rem;
            color: rgba(255,255,255,0.38);
            margin-bottom: 16px;
        }
        .ag-footer-chips {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 7px;
        }
        .ag-footer-chip {
            background: rgba(76,175,80,0.06);
            border: 1px solid rgba(76,175,80,0.13);
            border-radius: 999px;
            padding: 4px 12px;
            font-size: 0.72rem;
            color: rgba(255,255,255,0.50);
            font-weight: 500;
        }

        /* ── Mobile ───────────────────────────────────────────────────── */
        @media (max-width: 640px) {
            .hero-banner        { padding: 30px 18px 26px; }
            .hero-logo          { font-size: 2.1rem; }
            .hero-tagline       { font-size: 0.94rem; }
            .hero-stats         { gap: 10px; }
            .hero-stat          { min-width: 78px; padding: 11px 13px; }
            .hero-stat-val      { font-size: 1.15rem; }
            .ag-card            { padding: 18px 14px; border-radius: 16px; }
            .metric-grid        { grid-template-columns: repeat(2, 1fr); gap: 10px; }
            .advisor-grid       { grid-template-columns: 1fr; }
            .camera-card        { padding: 18px 14px; }
            .camera-card-heading{ font-size: clamp(1.2rem, 7vw, 1.6rem); }
            .section-title      { font-size: 1.1rem; }
        }
        @media (max-width: 400px) {
            .hero-powered       { gap: 6px; }
            .hero-badge         { font-size: 0.74rem; padding: 6px 11px; }
            .metric-grid        { gap: 8px; }
        }
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
# Helpers  (ALL LOGIC UNCHANGED)
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
    status  = get_provider_status()
    current = get_current_provider()

    with st.sidebar:
        st.markdown(
            """
            <div class="sb-logo"> AgriGuard AI</div>
            <div class="sb-version">v2.0 · Sustainable Agriculture Platform</div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        st.subheader(" Platform Overview")
        c1, c2 = st.columns(2)
        c1.metric("Model",   crop_config.get("model_name", "MobileNetV2"))
        c2.metric("Crop",    crop_config.get("crop_name",  "Tomato"))
        c1.metric("AI",      "Gemini + Groq")
        c2.metric("Version", "2.0")

        st.divider()
        st.subheader(" AI Provider Status")

        if current == PROVIDER_GEMINI:
            st.markdown('<div class="sb-pill g"> Active: Gemini 2.5 Flash</div>',
                        unsafe_allow_html=True)
        elif current == PROVIDER_GROQ:
            st.markdown('<div class="sb-pill q"> Active: Groq Llama 3.3 70B</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="sb-pill l"> Active: Local Knowledge Base</div>',
                        unsafe_allow_html=True)

        st.write("")
        if status["gemini_available"]:
            st.write(" Gemini 2.5 Flash — ready")
        else:
            st.write(" Gemini 2.5 Flash — unavailable")
            st.caption(status["gemini_error"])

        if status["groq_available"]:
            st.write(" Groq Llama 3.3 70B — ready")
        else:
            st.write(" Groq Llama 3.3 70B — unavailable")
            st.caption(status["groq_error"])

        st.write(" Local Knowledge Base — always ready")
        st.divider()
        st.caption("Priority: Gemini → Groq → Local KB. Keys loaded from `.env`.")

        # ── Validation analytics ──────────────────────────────────────────
        st.divider()
        st.subheader(" Validation Analytics")
        _init_validation_counters()
        gcalls  = st.session_state.get("gemini_validation_calls", 0)
        lpasses = st.session_state.get("local_validation_passes", 0)
        total   = gcalls + lpasses
        st.metric("Local Validation Passes", lpasses,
                  help="Images that skipped Gemini Vision (confidence >=95%)")
        st.metric("Gemini Validation Calls", gcalls,
                  help="Images that triggered Gemini Vision (confidence <95%)")
        if total > 0:
            saved_pct = round(lpasses / total * 100)
            st.progress(lpasses / total)
            st.caption(
                f"**{saved_pct}%** of images validated locally — "
                f"{lpasses} Gemini call(s) saved"
            )
        else:
            st.caption("No images validated yet in this session.")

        st.divider()
        st.subheader(" Built For")
        st.markdown(
            """
            <div>
              <span class="sb-chip"> 1M1B AI for Sustainability</span>
              <span class="sb-chip"> College Major Project</span>
              <span class="sb-chip"> Placement Portfolio</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Hero  (replaces render_header)
# ---------------------------------------------------------------------------
def render_hero(crop_config):
    active_provider = get_current_provider()
    status_text     = get_gemini_status()
    num_classes     = len(crop_config["class_names"])
    model_name      = crop_config.get("model_name", "MobileNetV2")
    crop_name       = crop_config.get("crop_name",  "Tomato")

    # First word only for the stat card (keeps it readable)
    provider_short = active_provider.split()[0] if active_provider else "—"

    if active_provider == PROVIDER_GEMINI:
        pill_cls, pill_icon = "gemini", ""
    elif active_provider == PROVIDER_GROQ:
        pill_cls, pill_icon = "groq",   ""
    else:
        pill_cls, pill_icon = "local",  ""

    st.markdown(
        f"""
        <div class="hero-banner">
          <p class="hero-eyebrow"> AI-Powered AgriTech Platform</p>
          <h1 class="hero-logo">AgriGuard AI</h1>
          <p class="hero-tagline">
            Smart Crop Disease Detection &amp;<br>
            Sustainable Farming Advisor
          </p>
          <div class="hero-powered">
            <span class="hero-badge">
              <span class="hero-dot"></span>MobileNetV2
            </span>
            <span class="hero-badge">
              <span class="hero-dot"></span>Gemini AI
            </span>
            <span class="hero-badge">
              <span class="hero-dot"></span>Sustainable Agriculture Intelligence
            </span>
          </div>
          <div class="hero-stats">
            <div class="hero-stat">
              <span class="hero-stat-val">{num_classes}</span>
              <span class="hero-stat-lbl">Disease Classes</span>
            </div>
            <div class="hero-stat">
              <span class="hero-stat-val">{_e(model_name)}</span>
              <span class="hero-stat-lbl">Model</span>
            </div>
            <div class="hero-stat">
              <span class="hero-stat-val">{_e(crop_name)}</span>
              <span class="hero-stat-lbl">Crop</span>
            </div>
            <div class="hero-stat">
              <span class="hero-stat-val">{_e(provider_short)}</span>
              <span class="hero-stat-lbl">AI Advisor</span>
            </div>
          </div>
          <div class="provider-pill {pill_cls}">
            {pill_icon} {_e(status_text)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Section 1 — Smart Leaf Scanner
# ---------------------------------------------------------------------------
def render_image_source_selector():
    # ── Section card header ───────────────────────────────────────────────
    st.markdown(
        """
        <div class="ag-card">
          <div class="section-header">
            <div class="section-header-text">
              <p class="section-title"> Smart Leaf Scanner</p>
              <p class="section-desc">
                Upload or capture a tomato leaf image for instant AI-powered disease diagnosis.
              </p>
            </div>
            <span class="section-badge"> Upload  Camera  Mobile</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Source toggle ─────────────────────────────────────────────────────
    source = st.radio(
        "Image source",
        [" Upload Image", " Capture from Camera"],
        horizontal=True,
        label_visibility="collapsed",
    )

    # ── Upload branch ─────────────────────────────────────────────────────
    if source == " Upload Image":
        st.markdown(
            """
            <div class="camera-card">
              <p class="camera-card-heading"> Upload Tomato Leaf</p>
              <p class="camera-card-subtitle">
                Select a JPG / PNG photo of a tomato leaf from your device
              </p>
              <div class="camera-instructions">
                <span class="camera-tip">
                  <span class="camera-tip-icon"></span> Use good lighting
                </span>
                <span class="camera-tip">
                  <span class="camera-tip-icon"></span> Tomato leaves only
                </span>
                <span class="camera-tip">
                  <span class="camera-tip-icon"></span> Avoid blurry images
                </span>
                <span class="camera-tip">
                  <span class="camera-tip-icon"></span> Leaf fills the frame
                </span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        image_file = st.file_uploader(
            "Upload Tomato Leaf Image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
            label_visibility="collapsed",
        )
        if image_file is None:
            st.info(
                "** For best results**  \n"
                "- Use natural or bright indoor light  \n"
                "- Leaf should fill most of the frame  \n"
                "- Avoid blurry or very dark images  \n"
                "- Tomato leaves only — other plants are rejected"
            )
        return image_file, "Uploaded"

    # ── Camera branch — mobile-first, full-width ──────────────────────────
    st.markdown(
        """
        <div class="camera-card">
          <p class="camera-card-heading"> Capture Tomato Leaf</p>
          <p class="camera-card-subtitle">
            Position the leaf in the frame, then tap the shutter button
          </p>
          <div class="camera-instructions">
            <span class="camera-tip">
              <span class="camera-tip-icon"></span> Hold leaf steady
            </span>
            <span class="camera-tip">
              <span class="camera-tip-icon"></span> Ensure good lighting
            </span>
            <span class="camera-tip">
              <span class="camera-tip-icon"></span> Keep leaf centered
            </span>
            <span class="camera-tip">
              <span class="camera-tip-icon"></span> Avoid blurry images
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Full-width camera widget (CSS forces 100 % width)
    image_file = st.camera_input(
        "Capture Tomato Leaf Image",
        label_visibility="collapsed",
        key="camera_capture",
    )

    # ── Post-capture UI ───────────────────────────────────────────────────
    if image_file is not None:
        st.success(" Photo captured! Review below, then choose an action.")
        st.image(
            image_file,
            caption=" Captured Leaf — Ready for Analysis",
            use_container_width=True,
        )
        st.write("")  # breathing room

        col_retake, col_analyze = st.columns(2, gap="small")
        with col_retake:
            if st.button(
                " Retake Photo",
                use_container_width=True,
                help="Discard this photo and capture again",
                key="btn_retake",
            ):
                st.session_state.pop("camera_capture", None)
                st.rerun()
        with col_analyze:
            st.button(
                " Analyze Leaf",
                use_container_width=True,
                type="primary",
                help="AI diagnosis runs automatically after capture",
                key="btn_analyze",
                disabled=True,   # visual affordance — analysis starts automatically
            )
        st.caption(
            "_Analysis starts automatically once you capture a photo. "
            "Tap **Retake Photo** if you want a better shot._"
        )

    return image_file, "Captured"


# ---------------------------------------------------------------------------
# Validation — confidence-gated Gemini Vision  (LOGIC UNCHANGED)
# ---------------------------------------------------------------------------
def render_validation_error(message: str):
    st.error(
        f" **Invalid Image**\n\n{message}\n\n"
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

    # ── Stage 1: Image quality ─────────────────────────────────────────────
    quality_ok, quality_msg = validate_image_quality(image)
    if not quality_ok:
        render_validation_error(quality_msg)
        st.stop()

    # ── Stage 2: Confidence-gated decision ────────────────────────────────
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
        with st.spinner(" Low confidence detected — verifying with Gemini Vision..."):
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
# Section 3 — AI Agronomist Advisor
# ---------------------------------------------------------------------------
def render_advisor_section(advisor_data: dict, advisor_source: str, advisor_provider: str):
    if advisor_source == "gemini":
        src_badge = '<span class="section-badge"> Gemini 2.5 Flash</span>'
    elif advisor_source == "groq":
        src_badge = f'<span class="section-badge"> {_e(advisor_provider)}</span>'
    else:
        src_badge = '<span class="section-badge"> Local Knowledge Base</span>'

    st.markdown(
        f"""
        <div class="ag-card">
          <div class="section-header">
            <div class="section-header-text">
              <p class="section-title"> AI Agronomist Advisor</p>
              <p class="section-desc">
                AI-generated advisory tailored to the detected disease.
                Always verify with local agricultural experts.
              </p>
            </div>
            {src_badge}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = [
        ("", "Farmer Explanation",         advisor_data.get("farmer_explanation", "—")),
        ("", "Prevention Measures",        advisor_data.get("prevention",         "—")),
        ("", "Treatment Strategy",         advisor_data.get("treatment_strategy", "—")),
        ("", "Long-Term Recommendations",  advisor_data.get("long_term_advice",   "—")),
        ("", "Sustainability Practices",   advisor_data.get("sustainability",     "—")),
    ]

    cards_html = '<div class="advisor-grid">'
    for icon, title, body in cards:
        cards_html += f"""
        <div class="advisor-card">
          <div class="advisor-card-title">{icon} {_e(title)}</div>
          <div class="advisor-card-body">{_e(body)}</div>
        </div>"""
    cards_html += "</div>"

    st.markdown(cards_html, unsafe_allow_html=True)
    st.write("")  # spacing after grid


# ---------------------------------------------------------------------------
# Section 6 — Reports & Export
# ---------------------------------------------------------------------------
def render_reports_section(report_data: dict, pdf_report: bytes):
    st.markdown(
        """
        <div class="ag-card">
          <div class="section-header">
            <div class="section-header-text">
              <p class="section-title"> Reports &amp; Export</p>
              <p class="section-desc">
                Download a professionally formatted PDF with all findings,
                treatment plans, and AI recommendations.
              </p>
            </div>
            <span class="section-badge"> PDF Report</span>
          </div>
          <div class="report-card">
            <p class="report-card-title"> Farmer Diagnosis Report</p>
            <p class="report-card-desc">
              Complete diagnosis &nbsp;·&nbsp; Treatment plan &nbsp;·&nbsp;
              AI advisor insights &nbsp;·&nbsp; Sustainability score
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.download_button(
        label=" Export Farmer Report",
        data=pdf_report,
        file_name=f"{report_data['report_id']}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Section 2 — AI Diagnosis Dashboard
# ---------------------------------------------------------------------------
def render_prediction_results(image, source_label, result, context, crop_config, image_fingerprint):
    severity = context["severity"]
    disease  = context["display_name"]
    conf     = context["confidence"]
    sust     = context["sustainability_score"]

    # Severity → CSS class + icon
    _sev_map = {
        "High":   ("high",    ""),
        "Medium": ("medium",  ""),
        "Low":    ("low",     ""),
        "None":   ("healthy", ""),
    }
    sev_cls, sev_icon = _sev_map.get(severity, ("info", ""))

    # Sustainability class
    sust_cls = "healthy" if sust >= 70 else "low" if sust >= 50 else "medium"

    # ── Section card header ───────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="ag-card">
          <div class="section-header">
            <div class="section-header-text">
              <p class="section-title"> AI Diagnosis Dashboard</p>
              <p class="section-desc">
                Real-time disease detection powered by MobileNetV2 + Gemini AI
              </p>
            </div>
            <span class="sev-badge {sev_cls}" style="flex-shrink:0">
              {sev_icon} {_e(severity)}
            </span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 4-card summary metric row ─────────────────────────────────────────
    disease_short = disease if len(disease) <= 20 else disease[:18] + "…"

    vsource = st.session_state.get("last_validation_source", "")
    if vsource == "local":
        vsource_html = '<span class="vsource local"> Local Validation (Gemini skipped)</span>'
    elif vsource == "gemini":
        vsource_html = '<span class="vsource gemini"> Gemini Vision Verified</span>'
    elif vsource == "gemini_fallback":
        vsource_html = '<span class="vsource fb"> AI Validation Unavailable</span>'
    else:
        vsource_html = ""

    st.markdown(
        f"""
        <div class="metric-grid">
          <div class="metric-card {sev_cls}">
            <span class="mc-icon"></span>
            <span class="mc-val">{_e(disease_short)}</span>
            <span class="mc-lbl">Detected Disease</span>
          </div>
          <div class="metric-card info">
            <span class="mc-icon"></span>
            <span class="mc-val">{conf:.1f}%</span>
            <span class="mc-lbl">AI Confidence</span>
          </div>
          <div class="metric-card {sev_cls}">
            <span class="mc-icon">{sev_icon}</span>
            <span class="mc-val">{_e(severity)}</span>
            <span class="mc-lbl">Severity Level</span>
          </div>
          <div class="metric-card {sust_cls}">
            <span class="mc-icon"></span>
            <span class="mc-val">{sust}/100</span>
            <span class="mc-lbl">Sustainability</span>
          </div>
        </div>
        {vsource_html}
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # ── Image + Top-5 Predictions (2 columns) ─────────────────────────────
    img_col, pred_col = st.columns([1, 1])

    with img_col:
        st.image(image, caption=f" {source_label} Leaf Image", use_container_width=True)

    with pred_col:
        st.subheader(" Top 5 Predictions")
        top_table = build_top_prediction_table(result, crop_config)
        for _, row in top_table.iterrows():
            st.write(f"**{row['Disease']}** — {row['Confidence']:.1f}%")
            st.progress(row["Confidence"] / 100)

        st.write("")
        st.subheader(" Confidence Comparison")
        st.bar_chart(top_table.set_index("Disease"))

    with st.expander(" Full Probability Table"):
        prob_table = build_probability_table(result["probabilities"], crop_config)
        st.dataframe(
            prob_table.sort_values(by="Probability (%)", ascending=False),
            use_container_width=True,
        )

    # ── Disease Details ───────────────────────────────────────────────────
    st.markdown(
        """
        <div class="ag-card">
          <div class="section-header">
            <div class="section-header-text">
              <p class="section-title"> Disease Details</p>
              <p class="section-desc">Comprehensive breakdown of the detected condition</p>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    info_col, action_col = st.columns(2)

    with info_col:
        st.markdown(
            f"""
            <div class="detail-card">
              <div class="detail-card-title"> Disease Description</div>
              <div class="detail-card-body">{_e(context['description'])}</div>
            </div>
            <div class="detail-card">
              <div class="detail-card-title"> Cause</div>
              <div class="detail-card-body">{_e(context['cause'])}</div>
            </div>
            <div class="detail-card">
              <div class="detail-card-title"> Symptoms</div>
              <div class="detail-card-body">{_e(context['symptoms'])}</div>
            </div>
            <div class="detail-card">
              <div class="detail-card-title"> Suggested Medication</div>
              <div class="detail-card-body">{_e(context['medication'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with action_col:
        st.markdown(
            f"""
            <div class="detail-card">
              <div class="detail-card-title"> Severity Level</div>
              <div class="detail-card-body">
                <span class="sev-badge {sev_cls}">{sev_icon} {_e(severity)}</span>
              </div>
            </div>
            <div class="detail-card">
              <div class="detail-card-title"> Treatment Plan</div>
              <div class="detail-card-body">{_e(context['treatment'])}</div>
            </div>
            <div class="detail-card">
              <div class="detail-card-title"> Sustainability Advice</div>
              <div class="detail-card-body">{_e(context['sustainability'])}</div>
            </div>
            <div class="detail-card">
              <div class="detail-card-title"> Prevention Tips</div>
              <div class="detail-card-body">{_e(context['prevention'])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── AI Advisor (generate + render as Section 3) ───────────────────────
    with st.spinner(" Generating AI recommendations..."):
        advisor_data = generate_advisory(
            disease=context["predicted_class"],
            cause=context["cause"],
            symptoms=context["symptoms"],
            medication=context["medication"],
        )

    advisor_source   = advisor_data.get("source",   "local")
    advisor_provider = advisor_data.get("provider", get_current_provider())

    # Store for chatbot context
    st.session_state["detected_disease"]         = context["predicted_class"]
    st.session_state["detected_disease_display"] = context["display_name"]

    render_advisor_section(advisor_data, advisor_source, advisor_provider)

    # ── Reports (Section 6) ───────────────────────────────────────────────
    report_data = build_report_data(context, image_fingerprint, advisor_data)
    pdf_report  = generate_pdf_report(report_data)
    render_reports_section(report_data, pdf_report)


# ---------------------------------------------------------------------------
# Section 4 — AI Farming Assistant Chatbot
# ---------------------------------------------------------------------------
def render_chatbot_section():
    current_provider = get_current_provider()

    if current_provider == PROVIDER_GEMINI:
        cp_cls, cp_icon = "gemini", ""
        cp_text = "Gemini 2.5 Flash Active"
    elif current_provider == PROVIDER_GROQ:
        cp_cls, cp_icon = "groq", ""
        cp_text = "Groq Llama 3.3 70B Active (Gemini fallback)"
    else:
        cp_cls, cp_icon = "local", ""
        cp_text = "Local Advisor Active — add API keys to enable cloud AI"

    detected_disease = st.session_state.get("detected_disease", "")
    detected_display = st.session_state.get("detected_disease_display", "")

    context_badge = (
        f'<span class="chat-context"> Context: {_e(detected_display)}</span>'
        if detected_display else ""
    )

    st.markdown(
        f"""
        <div class="ag-card">
          <div class="section-header">
            <div class="section-header-text">
              <p class="section-title"> AI Farming Assistant</p>
              <p class="section-desc">
                Ask about disease prevention, treatment, fertilizers,
                and sustainable tomato farming
              </p>
            </div>
          </div>
          <div>
            <span class="chat-provider {cp_cls}">{cp_icon} {_e(cp_text)}</span>
            {context_badge}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not detected_display:
        st.caption(
            "Upload a leaf image first to unlock disease-specific advice. "
            "General farming questions welcome."
        )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input(
        "Ask about disease prevention, treatment, fungicides, or sustainable farming…"
    )

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state["chat_history"].append({"role": "user", "content": user_question})

        with st.spinner(" AgriGuard AI is thinking..."):
            reply = chat_with_advisor(
                user_question=user_question,
                detected_disease=detected_disease,
                chat_history=st.session_state["chat_history"],
            )

        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})


# ---------------------------------------------------------------------------
# Section 5 — Farm Analytics Dashboard
# ---------------------------------------------------------------------------
def render_analytics_section(history_df):
    st.markdown(
        """
        <div class="ag-card">
          <div class="section-header">
            <div class="section-header-text">
              <p class="section-title"> Farm Analytics Dashboard</p>
              <p class="section-desc">
                Prediction history, disease distribution, confidence trends,
                and validation statistics
              </p>
            </div>
            <span class="section-badge"> Live Data</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Validation stats summary
    _init_validation_counters()
    gcalls  = st.session_state.get("gemini_validation_calls", 0)
    lpasses = st.session_state.get("local_validation_passes", 0)
    total   = gcalls + lpasses

    v1, v2, v3 = st.columns(3)
    v1.metric("Gemini Validation Calls",  gcalls,
              help="Images verified by Gemini Vision API")
    v2.metric("Local Validation Passes",  lpasses,
              help="Images that skipped Gemini (confidence >=95%)")
    if total > 0:
        v3.metric("API Savings", f"{round(lpasses / total * 100)}%",
                  help="Percentage of validations handled locally")
    else:
        v3.metric("API Savings", "—")

    render_analytics_dashboard(history_df)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
def render_footer():
    st.markdown(
        """
        <div class="ag-footer">
          <div class="ag-footer-logo"> AgriGuard AI</div>
          <div class="ag-footer-tagline">
            AI-Powered Sustainable Agriculture Platform · v2.0
          </div>
          <div class="ag-footer-chips">
            <span class="ag-footer-chip">TensorFlow</span>
            <span class="ag-footer-chip">MobileNetV2</span>
            <span class="ag-footer-chip">Streamlit</span>
            <span class="ag-footer-chip">Gemini AI</span>
            <span class="ag-footer-chip">Groq</span>
            <span class="ag-footer-chip">Python</span>
            <span class="ag-footer-chip">1M1B AI for Sustainability</span>
            <span class="ag-footer-chip">College Major Project</span>
            <span class="ag-footer-chip">Placement Portfolio</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    inject_css()

    crop_config = get_crop_config(ACTIVE_CROP)
    history_df  = load_history()

    render_sidebar(crop_config)
    render_hero(crop_config)           # Section 0: Hero

    try:
        trained_model = load_trained_model(crop_config["model_path"])
    except Exception as exc:
        st.error(
            f"Model could not be loaded from `{crop_config['model_path']}`. "
            "Please verify the model file exists."
        )
        st.caption(str(exc))
        render_analytics_section(history_df)
        return

    image_file, source_label = render_image_source_selector()   # Section 1
    image, image_fingerprint = load_leaf_image(image_file)

    if image is None:
        st.info(" Upload or capture a tomato leaf image above to begin AI diagnosis.")
    else:
        # ── Step 1: Local quality check (blur / brightness / resolution) ──
        quality_ok, quality_msg = validate_image_quality(image)
        if not quality_ok:
            render_validation_error(quality_msg)
            st.stop()

        # ── Step 2: Disease prediction ────────────────────────────────────
        try:
            with st.spinner(" Analysing leaf for disease patterns..."):
                result = predict_disease(image, trained_model, crop_config)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
        else:
            confidence = result["confidence"]

            # ── Step 3: Confidence-gated Gemini Vision validation ─────────
            # run_validation() returns (passed, source); calls st.stop() on failure
            _, vsource = run_validation(image, image_fingerprint, confidence)
            st.session_state["last_validation_source"] = vsource

            context    = get_prediction_context(result, crop_config)
            history_df = log_prediction_once(history_df, image_fingerprint, context)

            # Sections 2 + 3 + 6 (diagnosis, advisor, reports)
            render_prediction_results(
                image, source_label, result, context, crop_config, image_fingerprint
            )

    render_analytics_section(history_df)   # Section 5
    render_chatbot_section()               # Section 4
    render_footer()


if __name__ == "__main__":
    main()
