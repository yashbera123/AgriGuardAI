from fastapi import APIRouter, HTTPException, status

from api.schemas.history import AnalyticsResponse, HistoryResponse
from services.history_service import HistoryService
from utils.exceptions import DatabaseError
from utils.logger_config import get_logger

logger = get_logger(__name__)
# Removed prefix to allow both /history and /analytics
router = APIRouter(tags=["History", "Analytics"])
history_service = HistoryService()


@router.get(
    "/history", response_model=HistoryResponse, summary="Retrieve prediction history"
)
async def get_history():
    """
    Fetch the historical record of all disease predictions stored in the MySQL database.
    """
    try:
        records = history_service.get_history()

        formatted_records = []
        for r in records:
            formatted_records.append(
                {
                    "id": r.get("id", 0),
                    "disease_name": r.get("disease_name", "Unknown"),
                    "confidence": float(r.get("confidence", 0.0)),
                    "sustainability_score": float(r.get("sustainability_score", 0.0)),
                    "timestamp": str(r.get("timestamp", "")),
                }
            )

        return HistoryResponse(
            success=True,
            total_records=len(formatted_records),
            history=formatted_records,
        )
    except DatabaseError as e:
        logger.error(f"Database error fetching history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch prediction history",
        )
    except Exception as e:
        logger.error(f"Unexpected error in history retrieval: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.get(
    "/analytics", response_model=AnalyticsResponse, summary="Get prediction analytics"
)
async def get_analytics():
    """
    Fetch summarized analytics data based on historical predictions.
    """
    try:
        summary = history_service.get_analytics_summary()

        return AnalyticsResponse(
            success=True,
            total_predictions=summary.get("total_predictions", 0),
            data=summary.get("data", []),
        )
    except DatabaseError as e:
        logger.error(f"Database error fetching analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch analytics",
        )
    except Exception as e:
        logger.error(f"Unexpected error in analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
