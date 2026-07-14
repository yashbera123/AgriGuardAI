from fastapi import FastAPI
from pydantic import BaseModel

from api.routers import advisor, history, predict
from database.database import Database
from services.advisor_service import AdvisorService
from utils.exceptions import DatabaseError
from utils.logger_config import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="AgriGuard API",
    description="REST API for AgriGuard AI - Sustainable Agriculture Decision Support System",
    version="1.0.0",
)

app.include_router(predict.router)
app.include_router(advisor.router)
app.include_router(history.router)


class HealthResponse(BaseModel):
    status: str
    version: str
    database_connection: str
    ai_provider_status: dict


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def check_health():
    """
    Check the system health status.

    Returns the status of the API, current version, database connectivity,
    and the status of the configured AI providers.
    """
    db_status = "connected"
    try:
        db = Database()
        conn = db.get_connection()
        if conn:
            conn.close()
    except DatabaseError as e:
        logger.error(f"Health check DB error: {e}")
        db_status = "disconnected"

    try:
        advisor = AdvisorService()
        ai_status = advisor.check_provider_status()
    except Exception as e:
        logger.error(f"Health check AI error: {e}")
        ai_status = {"error": str(e)}

    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        version="1.0.0",
        database_connection=db_status,
        ai_provider_status=ai_status,
    )
