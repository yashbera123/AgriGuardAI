from fastapi import APIRouter, HTTPException, status

from api.schemas.advisor import (
    AdvisorRequest,
    AdvisorResponse,
    ChatRequest,
    ChatResponse,
)
from services.advisor_service import AdvisorService
from utils.exceptions import AdvisorError
from utils.logger_config import get_logger

logger = get_logger(__name__)
# The prefix is removed so we can define both /advisor and /chat paths explicitly
router = APIRouter(tags=["Advisor"])
advisor_service = AdvisorService()


@router.post(
    "/advisor",
    response_model=AdvisorResponse,
    summary="Get AI recommendation for a disease",
)
async def get_recommendation(request: AdvisorRequest):
    """
    Generate an AI agricultural advisory based on a disease prediction and confidence score.
    Uses the configured multi-provider LLM failover system.
    """
    try:
        advisory_text = advisor_service.get_advisory(
            request.disease_name, request.confidence
        )

        return AdvisorResponse(success=True, recommendation=advisory_text)
    except AdvisorError as e:
        logger.error(f"Advisor error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate advisory",
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/chat", response_model=ChatResponse, summary="Chat with AI Advisor")
async def chat_with_ai(request: ChatRequest):
    """
    Ask agricultural questions and get responses from the AI Advisor.
    Supports conversational history for context-aware answers.
    """
    try:
        response_text, provider_name = advisor_service.chat_with_ai(
            request.question, request.history
        )

        return ChatResponse(
            success=True, response=response_text, provider=provider_name
        )
    except AdvisorError as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat",
        )
    except Exception as e:
        logger.error(f"Unexpected error in chat: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
