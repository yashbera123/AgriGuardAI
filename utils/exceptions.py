"""Custom exceptions for AgriGuard AI."""


class AgriGuardError(Exception):
    """Base exception for all AgriGuard AI specific errors."""

    pass


class DatabaseError(AgriGuardError):
    """Raised when a database operation fails."""

    pass


class PredictionError(AgriGuardError):
    """Raised when the prediction model fails."""

    pass


class ValidationError(AgriGuardError):
    """Raised when input validation (like image quality) fails."""

    pass


class AdvisorError(AgriGuardError):
    """Raised when the AI advisor or LLM provider fails."""

    pass


class ReportGenerationError(AgriGuardError):
    """Raised when PDF report generation fails."""

    pass
