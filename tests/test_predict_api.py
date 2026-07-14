"""
tests/test_predict_api.py
=========================
Purpose
-------
Verify the behaviour of the POST /predict endpoint defined in api/routers/predict.py.

All ML inference is mocked to prevent loading TensorFlow and to ensure fast execution.

Test Matrix
-----------
  test_predict_accepts_valid_image               → HTTP 200, valid JPEG/PNG
  test_predict_response_contains_expected_fields → Validates response schema
  test_predict_rejects_invalid_file_type         → HTTP 400, e.g. text/plain
  test_predict_validation_error                  → HTTP 400, when validator fails
  test_predict_handles_prediction_error          → HTTP 500, when model fails
  test_predict_handles_unexpected_error          → HTTP 500, on generic exception
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from utils.exceptions import PredictionError, ValidationError

# ---------------------------------------------------------------------------
# Module-level TestClient
# ---------------------------------------------------------------------------
client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_image():
    """Returns a dummy byte sequence representing an image for upload."""
    return b"fake_image_content"


@pytest.fixture
def mock_validation():
    """Mocks the prediction_service.validate_image method."""
    with patch("api.routers.predict.prediction_service.validate_image") as mock_val:
        # Default happy-path: Image is valid
        mock_val.return_value = (True, "Valid tomato leaf image.")
        yield mock_val


@pytest.fixture
def mock_prediction():
    """Mocks the prediction_service.predict method."""
    with patch("api.routers.predict.prediction_service.predict") as mock_pred:
        # Default happy-path: Successful prediction
        mock_pred.return_value = {
            "disease_name": "Tomato_Early_blight",
            "confidence": 98.5,
            "class_idx": 1,
            "img_array": [[]],
        }
        yield mock_pred


# ---------------------------------------------------------------------------
# ── Happy-Path Tests ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestPredictHappyPath:
    """Verifies successful predictions with valid images."""

    def test_predict_accepts_valid_image(
        self, mock_validation, mock_prediction, dummy_image
    ):
        """A valid JPEG file should return HTTP 200."""
        response = client.post(
            "/predict", files={"image": ("test.jpg", dummy_image, "image/jpeg")}
        )
        assert response.status_code == 200

    def test_predict_response_contains_expected_fields(
        self, mock_validation, mock_prediction, dummy_image
    ):
        """The JSON response should contain prediction, confidence, timestamp, etc."""
        response = client.post(
            "/predict", files={"image": ("test.png", dummy_image, "image/png")}
        )
        data = response.json()

        # Verify schema
        assert data["success"] is True
        assert "prediction" in data
        assert data["prediction"] == "Tomato_Early_blight"

        assert "confidence" in data
        assert data["confidence"] == 98.5

        assert "timestamp" in data
        assert isinstance(data["timestamp"], str)

        assert "severity" in data
        assert "sustainability_score" in data


# ---------------------------------------------------------------------------
# ── Validation Failure Tests ────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestPredictValidation:
    """Verifies that invalid inputs are correctly rejected."""

    def test_predict_rejects_invalid_file_type(self, dummy_image):
        """Uploading a non-image file (e.g., text/plain) should return 400."""
        response = client.post(
            "/predict", files={"image": ("test.txt", dummy_image, "text/plain")}
        )
        assert response.status_code == 400
        assert "Invalid image type" in response.json()["detail"]

    def test_predict_validation_error(self, mock_validation, dummy_image):
        """When validate_image returns False, it should raise a 400 error."""
        # Override mock to simulate an invalid image (e.g., not a leaf)
        mock_validation.return_value = (False, "Image is not a tomato leaf.")

        response = client.post(
            "/predict", files={"image": ("test.jpg", dummy_image, "image/jpeg")}
        )
        assert response.status_code == 400
        assert response.json()["detail"] == "Image is not a tomato leaf."


# ---------------------------------------------------------------------------
# ── Server Error Tests ──────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestPredictServerErrors:
    """Verifies that exceptions during inference are handled correctly."""

    def test_predict_handles_prediction_error(
        self, mock_validation, mock_prediction, dummy_image
    ):
        """When the model raises PredictionError, it should return 500."""
        mock_prediction.side_effect = PredictionError("Inference failed")

        response = client.post(
            "/predict", files={"image": ("test.jpg", dummy_image, "image/jpeg")}
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "Prediction failed"

    def test_predict_handles_unexpected_error(
        self, mock_validation, mock_prediction, dummy_image
    ):
        """When an unhandled exception occurs, it should return 500."""
        mock_prediction.side_effect = Exception("Out of memory")

        response = client.post(
            "/predict", files={"image": ("test.jpg", dummy_image, "image/jpeg")}
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "Internal server error"
