from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    success: bool = Field(..., description="Whether the prediction was successful")
    prediction: str = Field(..., description="The predicted disease name")
    confidence: float = Field(..., description="Confidence score percentage (0-100)")
    severity: str = Field(..., description="Calculated severity of the disease")
    sustainability_score: int = Field(
        ..., description="Sustainability score out of 100"
    )
    timestamp: str = Field(..., description="ISO 8601 formatted timestamp")
    processing_time_ms: float = Field(
        ..., description="Time taken to process the image in milliseconds"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "prediction": "Tomato_Early_blight",
                "confidence": 98.5,
                "severity": "High",
                "sustainability_score": 90,
                "timestamp": "2026-07-12T17:24:42",
                "processing_time_ms": 150.5,
            }
        }
    }
