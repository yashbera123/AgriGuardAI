import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from config import MODEL_PATH

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    from utils.exceptions import PredictionError

    raise PredictionError(f"Failed to load model: {e}")

class_names = [
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

from utils.logger_config import get_logger

logger = get_logger(__name__)

img_path = input("Enter image path: ")

try:
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
except Exception as e:
    from utils.exceptions import PredictionError

    raise PredictionError(f"Prediction failed: {e}")

predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

logger.info(f"Prediction: {predicted_class}")
logger.info(f"Confidence: {confidence:.2f}%")
