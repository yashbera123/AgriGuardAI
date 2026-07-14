from datetime import datetime

from database.database import Database
from utils.logger_config import get_logger

logger = get_logger(__name__)


class PredictionRepository:
    """Repository for managing prediction history data."""

    def __init__(self):
        self.db = Database()
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Create the predictions table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME NOT NULL,
            disease_name VARCHAR(100) NOT NULL,
            confidence FLOAT NOT NULL,
            sustainability_score FLOAT NOT NULL
        )
        """
        self.db.execute_query(query)

    def save_prediction(
        self,
        disease_name: str,
        confidence: float,
        sustainability_score: float,
        timestamp: datetime = None,
    ) -> int:
        """Save a new prediction record."""
        logger.info(f"Saving prediction: {disease_name} ({confidence:.2f}%)")
        timestamp_str = (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        query = """
        INSERT INTO predictions (timestamp, disease_name, confidence, sustainability_score)
        VALUES (%s, %s, %s, %s)
        """
        params = (timestamp_str, disease_name, confidence, sustainability_score)
        return self.db.execute_query(query, params)

    def get_all_predictions(self) -> list:
        """Retrieve all prediction history."""
        logger.info("Retrieving all prediction history")
        query = "SELECT * FROM predictions ORDER BY timestamp ASC"
        return self.db.execute_query(query, fetch=True)
