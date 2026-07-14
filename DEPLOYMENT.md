# Deployment Guide (Render)

This document provides the necessary instructions and configurations to deploy AgriGuardAI to Render. The deployment is split into two Web Services: the FastAPI Backend and the Streamlit Frontend.

## 1. FastAPI Backend Service

**Service Type:** Web Service
**Environment:** Python

### Build Command
```bash
pip install -r requirements.txt
```

### Start Command
```bash
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

### Health Check
- **Path:** `/health`
- **Method:** `GET`

### Environment Variables
| Key | Value / Description |
|---|---|
| `DB_HOST` | Hostname of the external MySQL database (e.g., Railway) |
| `DB_USER` | MySQL Username |
| `DB_PASSWORD` | MySQL Password |
| `DB_NAME` | MySQL Database Name (`agriguard`) |
| `GEMINI_API_KEY` | Your Google Gemini API Key |
| `GROQ_API_KEY` | Your Groq Llama API Key |
| `PYTHONPATH` | `/opt/render/project/src` |

---

## 2. Streamlit Frontend Service

**Service Type:** Web Service
**Environment:** Python

### Build Command
```bash
pip install -r requirements.txt
```

### Start Command
```bash
streamlit run streamlit/app.py --server.port=$PORT --server.address=0.0.0.0
```

### Health Check
- **Path:** `/` (or Streamlit's native `/_stcore/health`)
- **Method:** `GET`

### Environment Variables
| Key | Value / Description |
|---|---|
| `API_URL` | The public URL of your deployed FastAPI Backend (e.g., `https://agriguard-api.onrender.com`) |
| `PYTHONPATH` | `/opt/render/project/src` |
