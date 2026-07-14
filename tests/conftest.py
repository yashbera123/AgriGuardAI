"""
tests/conftest.py
=================
Purpose
-------
Centralized pytest configuration and shared fixtures for AgriGuard AI.

This file ensures global test safety by automatically applying patches
that prevent accidental database connections and ML model loading
across the entire test suite.
"""

import os
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Global Environment Overrides
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def override_env_vars(monkeypatch):
    """
    Automatically mock environment variables for every test.
    Ensures tests don't inadvertently use production credentials.
    """
    monkeypatch.setenv("DB_HOST", "test_localhost")
    monkeypatch.setenv("DB_USER", "test_user")
    monkeypatch.setenv("DB_PASSWORD", "test_pass")
    monkeypatch.setenv("DB_NAME", "test_agriguard")

    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.setenv("GROQ_API_KEY", "test_groq_key")

    monkeypatch.setenv("MODEL_PATH", "dummy_path/test_model.keras")


# ---------------------------------------------------------------------------
# Global Security / Safety Hooks
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def block_real_mysql_connections():
    """
    Failsafe: Block the mysql.connector module globally during testing.
    If a test forgets to mock the Database class, it will crash here rather
    than connecting to a real local or production database.
    """
    with patch("mysql.connector.connect") as mock_connect:
        mock_connect.side_effect = Exception(
            "Security blocked: Attempted to connect to MySQL during a unit test. "
            "Please mock the database layer."
        )
        yield


# ---------------------------------------------------------------------------
# Shared Reusable Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_image_bytes():
    """
    Provides a standard dummy byte string representing an image.
    Can be reused by any test requiring file upload payloads or
    inference byte streams.
    """
    return b"test_fake_image_content"


@pytest.fixture
def mock_tf_keras():
    """
    Provides a reusable mock for tf.keras.models.load_model.
    Tests that need to instantiate the PredictionService but don't
    need specific model output can use this to bypass the slow TF load.
    """
    with patch("services.prediction_service.tf.keras.models.load_model") as mock_load:
        # Just return a dummy object
        mock_load.return_value = "dummy_model_instance"
        yield mock_load
