"""
tests/test_prediction_repository.py
===================================
Purpose
-------
Verify the behaviour of the PredictionRepository class in database/prediction_repository.py.

To ensure fast execution and avoid requiring a local MySQL instance, the
underlying Database class is mocked.

Test Matrix
-----------
  test_init_ensures_table_exists         → CREATE TABLE is executed on init
  test_save_prediction_success           → Correct query/params are passed for save
  test_save_prediction_with_timestamp    → Custom timestamps are handled correctly
  test_get_all_predictions_success       → History retrieval fetches and returns data
  test_database_error_propagation        → DatabaseError is passed upwards correctly
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from database.prediction_repository import PredictionRepository
from utils.exceptions import DatabaseError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db_class():
    """
    Mocks the Database class imported inside prediction_repository.
    Prevents the repository from attempting real MySQL connections.
    """
    with patch("database.prediction_repository.Database") as mock_db:
        mock_instance = MagicMock()
        mock_db.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def repo(mock_db_class):
    """Provides a PredictionRepository instance with a mocked Database."""
    return PredictionRepository()


# ---------------------------------------------------------------------------
# ── Initialization Tests ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestPredictionRepositoryInit:
    """Verifies repository setup logic."""

    def test_init_ensures_table_exists(self, mock_db_class):
        """When the repository is instantiated, it must ensure its table exists."""
        repo = PredictionRepository()

        # Verify execute_query was called during __init__
        mock_db_class.execute_query.assert_called_once()

        # Check that the query was a CREATE TABLE statement
        called_query = mock_db_class.execute_query.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS predictions" in called_query


# ---------------------------------------------------------------------------
# ── Save Prediction Tests ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestPredictionRepositorySave:
    """Verifies that predictions are saved with correct SQL parameters."""

    def test_save_prediction_success(self, repo, mock_db_class):
        """save_prediction should format arguments and execute an INSERT query."""
        # Reset mock to clear the __init__ call
        mock_db_class.reset_mock()

        # Simulate returning a new row ID of 42
        mock_db_class.execute_query.return_value = 42

        # Act
        inserted_id = repo.save_prediction(
            disease_name="Tomato_Early_blight",
            confidence=95.5,
            sustainability_score=80.0,
        )

        # Assert
        assert inserted_id == 42
        mock_db_class.execute_query.assert_called_once()

        args, kwargs = mock_db_class.execute_query.call_args
        query = args[0]
        params = args[1]

        assert "INSERT INTO predictions" in query
        assert params[1] == "Tomato_Early_blight"
        assert params[2] == 95.5
        assert params[3] == 80.0
        # params[0] is the auto-generated timestamp, we just check it's a string
        assert isinstance(params[0], str)

    def test_save_prediction_with_custom_timestamp(self, repo, mock_db_class):
        """save_prediction should respect custom timestamps if provided."""
        mock_db_class.reset_mock()
        custom_time = datetime(2026, 7, 13, 12, 0, 0)

        repo.save_prediction(
            disease_name="Healthy",
            confidence=100.0,
            sustainability_score=100.0,
            timestamp=custom_time,
        )

        params = mock_db_class.execute_query.call_args[0][1]
        assert params[0] == "2026-07-13 12:00:00"


# ---------------------------------------------------------------------------
# ── Retrieval Tests ─────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestPredictionRepositoryGet:
    """Verifies that history retrieval is executed correctly."""

    def test_get_all_predictions_success(self, repo, mock_db_class):
        """get_all_predictions should execute a SELECT and return the results."""
        mock_db_class.reset_mock()

        expected_results = [
            {"id": 1, "disease_name": "Tomato_Early_blight", "confidence": 95.0},
            {"id": 2, "disease_name": "Healthy", "confidence": 99.9},
        ]
        mock_db_class.execute_query.return_value = expected_results

        # Act
        results = repo.get_all_predictions()

        # Assert
        assert results == expected_results

        args, kwargs = mock_db_class.execute_query.call_args
        query = args[0]

        assert "SELECT * FROM predictions" in query
        assert kwargs.get("fetch") is True


# ---------------------------------------------------------------------------
# ── Error Handling Tests ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestPredictionRepositoryErrors:
    """Verifies that Database errors are properly bubbled up."""

    def test_database_error_propagation(self, repo, mock_db_class):
        """If the underlying DB raises DatabaseError, the repo must propagate it."""
        mock_db_class.reset_mock()
        mock_db_class.execute_query.side_effect = DatabaseError("Connection lost")

        with pytest.raises(DatabaseError) as exc_info:
            repo.get_all_predictions()

        assert "Connection lost" in str(exc_info.value)
