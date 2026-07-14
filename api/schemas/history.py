from typing import Any, List

from pydantic import BaseModel, Field


class HistoryRecord(BaseModel):
    id: int = Field(..., description="Unique ID of the prediction record")
    disease_name: str = Field(..., description="Predicted disease name")
    confidence: float = Field(..., description="Confidence score of prediction")
    sustainability_score: float = Field(
        ..., description="Calculated sustainability score"
    )
    timestamp: str = Field(..., description="When the prediction occurred")


class HistoryResponse(BaseModel):
    success: bool = Field(..., description="Whether history was retrieved successfully")
    total_records: int = Field(..., description="Total number of prediction records")
    history: List[HistoryRecord] = Field(..., description="List of prediction records")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "total_records": 1,
                "history": [
                    {
                        "id": 1,
                        "disease_name": "Tomato_Early_blight",
                        "confidence": 98.5,
                        "sustainability_score": 90.0,
                        "timestamp": "2026-07-12T17:24:42",
                    }
                ],
            }
        }
    }


class AnalyticsResponse(BaseModel):
    success: bool = Field(
        ..., description="Whether analytics were retrieved successfully"
    )
    total_predictions: int = Field(..., description="Total number of predictions made")
    data: list = Field(default=[], description="Raw data used for analytics charting")

    model_config = {
        "json_schema_extra": {
            "example": {"success": True, "total_predictions": 120, "data": []}
        }
    }
