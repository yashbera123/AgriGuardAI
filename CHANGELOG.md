# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-03-20

### Added
- **Disease Detection**: Integrated MobileNetV2 for rapid and accurate tomato leaf disease classification.
- **GradCAM**: Added Explainable AI (XAI) overlays to highlight diseased regions on the uploaded images.
- **AI Advisor**: Implemented Google Gemini (with Groq Llama 3 fallback) to provide intelligent agricultural guidance.
- **FastAPI**: Developed a highly scalable and decoupled REST API backend.
- **Docker**: Fully containerized the application (Streamlit, FastAPI, MySQL) with `docker-compose`.
- **MySQL**: Added robust database persistence for logging predictions and usage history.
- **Analytics**: Introduced a Streamlit analytics dashboard to view historical predictions and generate PDF reports.
