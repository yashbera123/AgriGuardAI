"""Advisor Service for AgriGuard AI."""

from services.gemini_advisor import chat_with_advisor, generate_advisory
from services.llm_manager import get_provider_status
from utils.exceptions import AdvisorError
from utils.logger_config import get_logger

logger = get_logger(__name__)


class AdvisorService:
    """Service layer for orchestrating LLM integrations and conversational AI."""

    def __init__(self):
        logger.info("Initializing AdvisorService")

    def get_advisory(self, disease_name: str, confidence: float) -> str:
        """Generate agricultural advisory using the best available provider."""
        logger.info(f"AdvisorService: getting advisory for {disease_name}")
        try:
            return generate_advisory(disease_name, confidence)
        except Exception as e:
            raise AdvisorError(f"Failed to generate advisory: {e}")

    def chat_with_ai(self, user_message: str, chat_history: list) -> tuple:
        """Process user message and generate a chatbot response.
        Returns a tuple of (response_text, provider_name)."""
        logger.info("AdvisorService: processing chat message")
        try:
            return chat_with_advisor(user_message, chat_history)
        except Exception as e:
            raise AdvisorError(f"Chat interaction failed: {e}")

    def check_provider_status(self) -> dict:
        """Check the status of available AI providers (Gemini, Groq, Local)."""
        logger.info("AdvisorService: checking provider status")
        try:
            # Assume get_provider_status returns some dict or state object
            return get_provider_status()
        except Exception as e:
            raise AdvisorError(f"Failed to check provider status: {e}")
