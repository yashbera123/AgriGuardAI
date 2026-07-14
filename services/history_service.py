"""History Service for AgriGuard AI."""

import pandas as pd

from database.prediction_repository import PredictionRepository
from utils.exceptions import DatabaseError
from utils.logger_config import get_logger

logger = get_logger(__name__)


class HistoryService:
    """Service layer for managing prediction histories and analytics preparation."""

    def __init__(self):
        self.repo = PredictionRepository()

    def save_prediction(
        self, disease_name: str, confidence: float, sustainability_score: float
    ) -> dict:
        """Save a prediction result to the history."""
        logger.info(f"HistoryService: saving prediction for {disease_name}")
        try:
            self.repo.save_prediction(disease_name, confidence, sustainability_score)
            return {"status": "success"}
        except Exception as e:
            raise DatabaseError(f"Failed to save prediction: {e}")

    def get_history(self) -> list:
        """Retrieve the prediction history."""
        logger.info("HistoryService: fetching history")
        try:
            return self.repo.get_all_predictions()
        except Exception as e:
            raise DatabaseError(f"Failed to fetch history: {e}")

    def get_analytics_summary(self) -> dict:
        """Process history data into analytics metrics."""
        # For simplicity, returning raw data to let analytics.py handle it, or we could do logic here.
        records = self.get_history()
        return {"total_predictions": len(records), "data": records}
