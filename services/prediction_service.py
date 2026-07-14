"""Prediction Service for AgriGuard AI."""

from io import BytesIO

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from config import MODEL_PATH
from utils.exceptions import PredictionError, ValidationError
from utils.gradcam import generate_gradcam_safe
from utils.leaf_validator import validate_image_quality, validate_tomato_leaf
from utils.logger_config import get_logger

logger = get_logger(__name__)


class PredictionService:
    """Service layer for handling model predictions and image validations."""

    def __init__(self):
        logger.info("Initializing PredictionService")
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            raise PredictionError(f"Failed to load model from {MODEL_PATH}: {e}")

        self.class_names = [
            "Tomato_Bacterial_spot",
            "Tomato_Early_blight",
            "Tomato_Late_blight",
            "Tomato_Leaf_Mold",
            "Tomato_Septoria_leaf_spot",
            "Tomato_Spider_mites_Two_spotted_spider_mite",
            "Tomato_Target_Spot",
            "Tomato_Tomato_YellowLeaf_Curl_Virus",
            "Tomato_Tomato_mosaic_virus",
            "Tomato_healthy",
        ]

    def validate_image(self, image_bytes: bytes) -> tuple[bool, str]:
        """Validate if the uploaded image is suitable for prediction."""
        logger.info("Validating uploaded image")
        is_valid_quality, quality_msg = validate_image_quality(image_bytes)
        if not is_valid_quality:
            return False, quality_msg

        is_valid_leaf, leaf_msg = validate_tomato_leaf(image_bytes)
        if not is_valid_leaf:
            return False, leaf_msg

        return True, "Valid tomato leaf image."

    def predict(self, image_bytes: bytes) -> dict:
        """Run inference on the provided image and return prediction details."""
        logger.info("Running disease prediction model")
        try:
            img = image.load_img(BytesIO(image_bytes), target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array_exp = np.expand_dims(img_array, axis=0)

            prediction = self.model.predict(img_array_exp)
            predicted_class_idx = int(np.argmax(prediction))
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(np.max(prediction) * 100)

            return {
                "disease_name": predicted_class,
                "confidence": confidence,
                "class_idx": predicted_class_idx,
                "img_array": img_array,
            }
        except Exception as e:
            raise PredictionError(f"Prediction inference failed: {e}")

    def generate_heatmap(self, img_array, predicted_class_idx):
        """Generate a GradCAM heatmap for explainability."""
        logger.info("Generating GradCAM heatmap")
        try:
            return generate_gradcam_safe(img_array, self.model, predicted_class_idx)
        except Exception as e:
            logger.error(f"Failed to generate heatmap: {e}")
            return None
