"""Prediction history storage for AgriGuard AI."""

from datetime import datetime
from pathlib import Path

import pandas as pd

from database.prediction_repository import PredictionRepository

HISTORY_PATH = Path("data") / "prediction_history.csv"
HISTORY_COLUMNS = [
    "Timestamp",
    "Disease Name",
    "Confidence",
    "Sustainability Score",
]


def load_history(history_path=HISTORY_PATH):
    """Load prediction history from the database, maintaining DataFrame compatibility."""
    repo = PredictionRepository()
    records = repo.get_all_predictions()

    if not records:
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    df = pd.DataFrame(records)

    mapping = {
        "timestamp": "Timestamp",
        "disease_name": "Disease Name",
        "confidence": "Confidence",
        "sustainability_score": "Sustainability Score",
    }
    df = df.rename(columns=mapping)

    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    for column in HISTORY_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    df = df[HISTORY_COLUMNS]
    df["Confidence"] = pd.to_numeric(
        df["Confidence"],
        errors="coerce",
    )
    df["Sustainability Score"] = pd.to_numeric(
        df["Sustainability Score"],
        errors="coerce",
    )

    return df


def append_prediction(
    disease_name,
    confidence,
    sustainability_score,
    timestamp=None,
    history_path=HISTORY_PATH,
):
    """Save one prediction to the database and return the updated history."""
    repo = PredictionRepository()

    dt_timestamp = None
    if timestamp:
        try:
            dt_timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            dt_timestamp = datetime.now()

    repo.save_prediction(
        disease_name=disease_name,
        confidence=float(confidence),
        sustainability_score=float(sustainability_score),
        timestamp=dt_timestamp,
    )

    return load_history(history_path)
