"""Verification script for gemini_advisor.py — safe for Windows cp1252 terminals."""
import ast
import os
import sys

# Force UTF-8 output on Windows to avoid cp1252 errors when printing results
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------- syntax ----------
ast.parse(open("gemini_advisor.py", encoding="utf-8").read())
print("[OK] gemini_advisor.py syntax valid")

ast.parse(open("app.py", encoding="utf-8").read())
print("[OK] app.py syntax valid")

# ---------- import (no real key — uses fallback) ----------
os.environ.pop("GEMINI_API_KEY", None)
from gemini_advisor import (
    chat_with_advisor,
    generate_advisory,
    get_gemini_status,
    is_gemini_available,
)
print("[OK] gemini_advisor imported")
print("[OK] is_gemini_available():", is_gemini_available())
print("[OK] get_gemini_status():", get_gemini_status())

# ---------- generate_advisory ----------
adv = generate_advisory("Tomato_Late_blight", cause="X", symptoms="Y", medication="Z")
assert adv.get("source") in ("gemini", "local"), "Bad source"
assert all(k in adv for k in ["farmer_explanation", "prevention", "sustainability", "treatment_strategy", "long_term_advice"])
print("[OK] generate_advisory source:", adv["source"])

# ---------- chatbot — no disease context ----------
reply_general = chat_with_advisor("Who are you?")
assert len(reply_general) > 10
print("[OK] chat (general question) length:", len(reply_general))

# ---------- chatbot — with disease context ----------
reply_disease = chat_with_advisor("How can I prevent this?", detected_disease="Tomato_Late_blight")
assert len(reply_disease) > 10
print("[OK] chat (disease context) length:", len(reply_disease))

# ---------- PDF still works ----------
from pdf_generator import generate_pdf_report
pdf = generate_pdf_report({
    "report_id": "TEST-001", "generated_at": "2026-06-20",
    "disease_name": "Late Blight", "confidence_score": "98%",
    "cause": "X", "symptoms": "Y", "medication": "Z",
    "treatment": "W", "sustainability_advice": "S",
    "sustainability_score": "40/100", "prevention_tips": "P",
    "advisor_data": adv,
})
size = len(pdf.read())
assert size > 1000
print("[OK] PDF with advisor section:", size, "bytes")

# ---------- packages ----------
for pkg in ["dotenv", "google.genai"]:
    try:
        __import__(pkg)
        print(f"[OK] {pkg} installed")
    except ImportError:
        print(f"[MISSING] {pkg} — run: pip install {pkg.replace('.', '-')}")

print()
print("=" * 40)
print("ALL CHECKS PASSED")
print("=" * 40)
