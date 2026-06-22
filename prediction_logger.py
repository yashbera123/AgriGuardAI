"""Prediction history storage for AgriGuard AI."""

from datetime import datetime
from pathlib import Path

import pandas as pd

HISTORY_PATH = Path("data") / "prediction_history.csv"
HISTORY_COLUMNS = [
    "Timestamp",
    "Disease Name",
    "Confidence",
    "Sustainability Score",
]


def ensure_history_file(history_path=HISTORY_PATH):
    """Create the history CSV safely when it does not already exist."""
    path = Path(history_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(path, index=False)

    return path


def load_history(history_path=HISTORY_PATH):
    """Load prediction history, repairing missing or empty files gracefully."""
    path = ensure_history_file(history_path)

    try:
        history = pd.read_csv(path)
    except (
        FileNotFoundError,
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ):
        history = pd.DataFrame(columns=HISTORY_COLUMNS)
        history.to_csv(path, index=False)

    for column in HISTORY_COLUMNS:
        if column not in history.columns:
            history[column] = pd.NA

    history = history[HISTORY_COLUMNS]
    history["Confidence"] = pd.to_numeric(
        history["Confidence"],
        errors="coerce",
    )
    history["Sustainability Score"] = pd.to_numeric(
        history["Sustainability Score"],
        errors="coerce",
    )

    return history


def append_prediction(
    disease_name,
    confidence,
    sustainability_score,
    timestamp=None,
    history_path=HISTORY_PATH,
):
    """Append one prediction to the CSV and return the updated history."""
    path = ensure_history_file(history_path)
    history = load_history(path)

    new_row = {
        "Timestamp": timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Disease Name": disease_name,
        "Confidence": round(float(confidence), 2),
        "Sustainability Score": round(float(sustainability_score), 2),
    }

    updated_history = pd.concat(
        [history, pd.DataFrame([new_row])],
        ignore_index=True,
    )
    updated_history.to_csv(path, index=False)

    return updated_history
