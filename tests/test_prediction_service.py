"""
tests/test_prediction_service.py
================================
Purpose
-------
Verify the behaviour of the PredictionService class in services/prediction_service.py.

To ensure fast execution and avoid TensorFlow overhead/errors during CI,
tf.keras.models.load_model and image processing functions are aggressively mocked.

Test Matrix
-----------
  test_init_loads_model_successfully      → Service initializes correctly
  test_init_handles_load_failure          → Raises PredictionError if model missing
  test_validate_image_success             → Returns True when validators pass
  test_validate_image_quality_failure     → Returns False when quality check fails
  test_validate_image_leaf_failure        → Returns False when leaf check fails
  test_predict_returns_expected_structure → Validates dictionary output format
  test_predict_handles_inference_failure  → Raises PredictionError on inference crash
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.prediction_service import PredictionService
from utils.exceptions import PredictionError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tf_model():
    """Mocks tf.keras.models.load_model to avoid actually loading a .keras file."""
    with patch("services.prediction_service.tf.keras.models.load_model") as mock_load:
        mock_model_instance = MagicMock()
        # Simulate model.predict returning a dummy probability array for 10 classes
        dummy_probs = np.zeros((1, 10))
        dummy_probs[0, 1] = 0.98  # Class index 1 ("Tomato_Early_blight") gets 98%
        mock_model_instance.predict.return_value = dummy_probs

        mock_load.return_value = mock_model_instance
        yield mock_load


@pytest.fixture
def prediction_service(mock_tf_model):
    """Provides a PredictionService instance with a mocked TensorFlow model."""
    return PredictionService()


@pytest.fixture
def dummy_image_bytes():
    return b"fake_image_bytes"


# ---------------------------------------------------------------------------
# ── Initialization Tests ────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestPredictionServiceInit:
    """Verifies the service handles model loading correctly."""

    def test_init_loads_model_successfully(self, mock_tf_model):
        """Service should initialize without errors if load_model succeeds."""
        service = PredictionService()
        assert service.model is not None
        assert len(service.class_names) == 10

    def test_init_handles_load_failure(self):
        """Service should raise PredictionError if load_model fails."""
        with patch(
            "services.prediction_service.tf.keras.models.load_model",
            side_effect=Exception("File not found"),
        ):
            with pytest.raises(PredictionError) as exc_info:
                PredictionService()
            assert "Failed to load model" in str(exc_info.value)


# ---------------------------------------------------------------------------
# ── Validation Tests ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestPredictionServiceValidation:
    """Verifies that the validation layer delegates correctly to leaf_validator."""

    @patch("services.prediction_service.validate_image_quality")
    @patch("services.prediction_service.validate_tomato_leaf")
    def test_validate_image_success(
        self, mock_leaf, mock_quality, prediction_service, dummy_image_bytes
    ):
        mock_quality.return_value = (True, "Quality OK")
        mock_leaf.return_value = (True, "Leaf OK")

        is_valid, msg = prediction_service.validate_image(dummy_image_bytes)

        assert is_valid is True
        assert msg == "Valid tomato leaf image."

    @patch("services.prediction_service.validate_image_quality")
    def test_validate_image_quality_failure(
        self, mock_quality, prediction_service, dummy_image_bytes
    ):
        mock_quality.return_value = (False, "Image too blurry")

        is_valid, msg = prediction_service.validate_image(dummy_image_bytes)

        assert is_valid is False
        assert msg == "Image too blurry"

    @patch("services.prediction_service.validate_image_quality")
    @patch("services.prediction_service.validate_tomato_leaf")
    def test_validate_image_leaf_failure(
        self, mock_leaf, mock_quality, prediction_service, dummy_image_bytes
    ):
        mock_quality.return_value = (True, "Quality OK")
        mock_leaf.return_value = (False, "Not a tomato leaf")

        is_valid, msg = prediction_service.validate_image(dummy_image_bytes)

        assert is_valid is False
        assert msg == "Not a tomato leaf"


# ---------------------------------------------------------------------------
# ── Prediction Tests ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestPredictionServiceInference:
    """Verifies the inference processing pipeline and return structures."""

    @patch("services.prediction_service.image")
    def test_predict_returns_expected_structure(
        self, mock_keras_image, prediction_service, dummy_image_bytes
    ):
        """
        predict() should preprocess the image, call model.predict, and format
        the results into a specific dictionary structure.
        """
        # Setup mocks for keras preprocessing
        mock_img = MagicMock()
        mock_keras_image.load_img.return_value = mock_img

        # Simulate img_to_array returning a dummy 224x224x3 numpy array
        dummy_array = np.zeros((224, 224, 3))
        mock_keras_image.img_to_array.return_value = dummy_array

        # Act
        result = prediction_service.predict(dummy_image_bytes)

        # Assert Expected Structure
        assert isinstance(result, dict)
        assert "disease_name" in result
        assert "confidence" in result
        assert "class_idx" in result
        assert "img_array" in result

        # Assert Expected Values (based on our mock_tf_model probability array)
        assert result["disease_name"] == "Tomato_Early_blight"
        assert result["class_idx"] == 1
        assert result["confidence"] == 98.0  # 0.98 * 100
        np.testing.assert_array_equal(result["img_array"], dummy_array)

    @patch("services.prediction_service.image")
    def test_predict_handles_inference_failure(
        self, mock_keras_image, prediction_service, dummy_image_bytes
    ):
        """If any part of inference fails, it must be wrapped in a PredictionError."""
        # Force preprocessing to fail
        mock_keras_image.load_img.side_effect = Exception("Corrupt image data")

        with pytest.raises(PredictionError) as exc_info:
            prediction_service.predict(dummy_image_bytes)

        assert "Prediction inference failed" in str(exc_info.value)
