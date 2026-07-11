# 🌿 AgriGuard AI
### AI-Powered Sustainable Crop Disease Detection & Decision Support System

AgriGuard AI is an intelligent crop disease diagnosis platform that combines **Deep Learning**, **Explainable AI**, **Large Language Models (LLMs)**, and **Sustainable Agriculture** to assist farmers in identifying tomato leaf diseases and receiving actionable treatment recommendations.

The system utilizes a **MobileNetV2 Convolutional Neural Network** for disease classification, integrates **Google Gemini 2.5 Flash** for intelligent agricultural advice, provides **Groq Llama 3.3** as an automatic fallback, and includes a **local knowledge base** to ensure uninterrupted operation even without cloud AI services.

---

# 📌 Table of Contents

- Project Overview
- Features
- System Architecture
- Technology Stack
- Project Structure
- Workflow
- Disease Classes
- Explainable AI
- Analytics Dashboard
- AI Advisor
- Chatbot
- PDF Report Generation
- Installation
- Running the Project
- Screenshots
- Future Scope
- Contributors
- License

---

# 🌱 Project Overview

Crop diseases significantly reduce agricultural productivity and often require expert diagnosis, which may not be readily available to every farmer.

AgriGuard AI addresses this challenge by providing an AI-powered diagnosis platform capable of:

- Detecting tomato leaf diseases
- Explaining predictions using Explainable AI
- Providing treatment recommendations
- Suggesting sustainable farming practices
- Answering farmer questions using AI
- Maintaining prediction history
- Generating professional PDF reports

The project was developed as a Final Year Major Project focusing on Sustainable Agriculture and Artificial Intelligence.

---

# 🚀 Features

## 🌿 Disease Detection

- MobileNetV2 CNN-based classifier
- 10 tomato leaf disease classes
- Confidence score for every prediction
- Top-5 prediction probabilities

---

## 📸 Image Validation

Before prediction the application verifies:

- Image quality
- Brightness
- Blur detection
- Tomato leaf validation

Invalid or unrelated images are rejected before reaching the model.

---

## 🧠 Explainable AI

The system supports Explainable AI through:

- Grad-CAM visualization
- Confidence comparison
- Probability distribution
- Top-5 prediction analysis

This helps users understand why the model predicted a particular disease.

---

## 🤖 AI Agronomist

AgriGuard AI generates intelligent recommendations using:

### Primary

Google Gemini 2.5 Flash

### Automatic Fallback

Groq Llama 3.3 70B

### Offline Fallback

Local Agricultural Knowledge Base

The AI provides:

- Disease explanation
- Prevention methods
- Treatment strategy
- Sustainable farming advice
- Long-term recommendations

---

## 💬 AI Farming Assistant

Interactive chatbot capable of answering questions related to:

- Disease prevention
- Crop protection
- Fungicides
- Sustainable farming
- Tomato cultivation

The chatbot automatically switches between:

Gemini → Groq → Local Knowledge Base

---

## 📊 Analytics Dashboard

The application records previous predictions and provides analytics including:

- Total predictions
- Most common disease
- Average confidence
- Average sustainability score
- Disease distribution
- Disease frequency
- Prediction history

---

## 📄 PDF Report Generation

Generate downloadable reports containing:

- Disease diagnosis
- Confidence score
- Cause
- Symptoms
- Medication
- Treatment plan
- Prevention tips
- Sustainability advice
- AI-generated recommendations
- Report ID
- Timestamp

---

# 🏗 System Architecture

```
                   User
                     │
                     ▼
            Streamlit Web Application
                     │
                     ▼
           Image Upload / Camera Capture
                     │
                     ▼
            Image Quality Validation
                     │
                     ▼
         Tomato Leaf Verification
                     │
                     ▼
         MobileNetV2 Disease Prediction
                     │
         ┌───────────┴────────────┐
         ▼                        ▼
   Grad-CAM Analysis      Probability Analysis
         │                        │
         └───────────┬────────────┘
                     ▼
          Disease Knowledge Base
                     │
                     ▼
              AI Agronomist
       Gemini → Groq → Local
                     │
                     ▼
      PDF Report + Analytics Dashboard
                     │
                     ▼
          AI Farming Assistant
```

---

# 🛠 Technology Stack

## Machine Learning

- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pandas

---

## Explainable AI

- Grad-CAM

---

## AI Models

- Google Gemini 2.5 Flash
- Groq Llama 3.3 70B
- Local Knowledge Base

---

## Frontend

- Streamlit
- HTML
- CSS
- Custom Glassmorphism UI

---

## Backend

- Python

---

## Visualization

- Matplotlib
- Streamlit Charts

---

## Report Generation

- ReportLab

---

## Image Processing

- Pillow
- OpenCV

---

# 📂 Project Structure

```
AgriGuardAI/

│── app.py
│── analytics.py
│── crop_knowledge.py
│── disease_info.py
│── recommendations.py
│── predict.py
│── gradcam.py
│── leaf_validator.py
│── gemini_advisor.py
│── llm_manager.py
│── pdf_generator.py
│── prediction_logger.py
│── train.py
│── requirements.txt
│
├── model/
├── dataset/
├── data/
├── screenshots/
├── .streamlit/
```

---

# 🔬 Disease Classes

The model can detect:

- Tomato Healthy
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites
- Target Spot
- Tomato Mosaic Virus
- Yellow Leaf Curl Virus

---

# 🔄 Workflow

```
Upload Image
      │
      ▼
Image Validation
      │
      ▼
Tomato Leaf Verification
      │
      ▼
MobileNetV2 Prediction
      │
      ▼
Grad-CAM Visualization
      │
      ▼
Knowledge Base
      │
      ▼
Gemini / Groq AI Advisor
      │
      ▼
Analytics Dashboard
      │
      ▼
PDF Report
      │
      ▼
AI Chatbot
```

---

# 📸 Screenshots

Include screenshots of:

- Home Page
- Upload Page
- Disease Prediction
- Grad-CAM Visualization
- AI Advisor
- Analytics Dashboard
- Chatbot
- PDF Report

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/AgriGuardAI.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```
GEMINI_API_KEY=YOUR_API_KEY
GROQ_API_KEY=YOUR_API_KEY
```

Run the application

```bash
streamlit run app.py
```

---

# 📈 Model Information

Model

- MobileNetV2

Input Size

- 224 × 224

Output Classes

- 10

Framework

- TensorFlow

---

# 🌍 Applications

- Smart Agriculture
- Precision Farming
- Crop Disease Detection
- Farmer Assistance
- Sustainable Agriculture
- AI-based Advisory Systems
- Agricultural Research

---

# 🚀 Future Scope

Future enhancements include:

- Multi-crop disease detection
- Cloud deployment on Microsoft Azure
- Real-time weather integration
- IoT sensor integration
- Mobile application
- Multi-language farmer support
- AI-powered disease progression prediction
- Farm management dashboard
- Satellite image analysis

---

# 👨‍💻 Developer

**Yashwant**

Final Year B.Tech Computer Science Engineering

AI • Machine Learning • Data Analytics • Cloud Computing

---

# 📜 License

This project is developed for educational and research purposes.
