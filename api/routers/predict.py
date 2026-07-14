import time
from datetime import datetime

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from api.schemas.prediction import PredictionResponse
from services.prediction_service import PredictionService
from streamlit_app.crop_knowledge import ACTIVE_CROP, get_crop_config
from streamlit_app.recommendations import recommendations
from utils.exceptions import PredictionError, ValidationError
from utils.logger_config import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction"])
prediction_service = PredictionService()

ALLOWED_TYPES = ["image/jpeg", "image/png"]
crop_config = get_crop_config(ACTIVE_CROP)


@router.post(
    "", response_model=PredictionResponse, summary="Predict disease from leaf image"
)
async def predict_disease(
    image: UploadFile = File(..., description="Tomato leaf image (JPEG/PNG)")
):
    """
    Upload a tomato leaf image to identify diseases and calculate severity and sustainability scores.
    """
    start_time = time.time()

    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image type. Allowed: {', '.join(ALLOWED_TYPES)}",
        )

    try:
        image_bytes = await image.read()

        is_valid, msg = prediction_service.validate_image(image_bytes)
        if not is_valid:
            raise ValidationError(msg)

        result = prediction_service.predict(image_bytes)
        disease_name = result.get("disease_name", "Unknown")

        # Calculate severity and sustainability
        sustainability_score = crop_config["sustainability_scores"].get(
            disease_name, 60
        )
        disease_data = recommendations.get(disease_name, {})
        severity = disease_data.get("severity", "Unknown")

        processing_time = (time.time() - start_time) * 1000

        return PredictionResponse(
            success=True,
            prediction=disease_name,
            confidence=result.get("confidence", 0.0),
            severity=severity,
            sustainability_score=sustainability_score,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(processing_time, 2),
        )

    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except PredictionError as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed",
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
