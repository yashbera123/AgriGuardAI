"""Report Service for AgriGuard AI."""

from database.report_repository import ReportRepository
from utils.exceptions import ReportGenerationError
from utils.logger_config import get_logger
from utils.pdf_generator import generate_pdf_report

logger = get_logger(__name__)


class ReportService:
    """Service layer for handling PDF generation logic."""

    def __init__(self):
        self.repo = ReportRepository()

    def generate_report(self, prediction_data: dict, advisory_text: str) -> str:
        """Generate a PDF report and return the file path or identifier."""
        logger.info("ReportService: generating PDF report")
        try:
            # We assume prediction_data contains the necessary keys
            pdf_path = generate_pdf_report(
                disease_name=prediction_data.get("disease_name", "Unknown"),
                confidence=prediction_data.get("confidence", 0.0),
                sustainability=prediction_data.get("sustainability", 0.0),
                recommendations=advisory_text,
                image_path=prediction_data.get("image_path"),
            )
            return pdf_path
        except Exception as e:
            raise ReportGenerationError(f"Failed to generate report: {e}")
