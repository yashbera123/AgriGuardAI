"""
tests/test_health_api.py
========================
Purpose
-------
Verify the behaviour of the GET /health endpoint defined in api/main.py.

All external I/O (database connections, AI provider calls) is mocked so
that these tests are:
  • Fast   – no network round-trips, no MySQL connections
  • Stable – results are deterministic regardless of environment
  • Safe   – the real TensorFlow model is never loaded

Test Matrix
-----------
  test_health_returns_http_200                   → HTTP status code
  test_health_response_schema_has_all_fields     → schema completeness
  test_health_field_types_are_correct            → type safety
  test_health_version_is_correct                 → version contract
  test_health_status_is_healthy_when_db_ok       → happy-path status
  test_health_ai_provider_status_is_returned     → AI data forwarded
  test_health_status_degraded_on_db_failure      → DB failure path
  test_health_db_disconnected_on_db_failure      → DB failure field
  test_health_ai_error_stored_in_response        → AI failure path
  test_health_still_200_when_ai_fails            → graceful AI failure
  test_health_both_db_and_ai_fail                → dual-failure path
  test_health_method_not_allowed_post            → HTTP contract
  test_health_method_not_allowed_put             → HTTP contract
  test_health_method_not_allowed_delete          → HTTP contract
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from utils.exceptions import DatabaseError

# ---------------------------------------------------------------------------
# Module-level TestClient
# A single shared client is fine for read-only, stateless endpoint tests.
# ---------------------------------------------------------------------------
client = TestClient(app)


# ---------------------------------------------------------------------------
# Reusable mock factories
# ---------------------------------------------------------------------------


def _make_db_mock(connected: bool = True, raise_error: bool = False) -> MagicMock:
    """
    Build a mock for the Database class.

    Args:
        connected:   Whether the mock connection object evaluates as truthy.
        raise_error: If True the mock raises DatabaseError on get_connection().
    """
    mock_db = MagicMock()
    if raise_error:
        mock_db.get_connection.side_effect = DatabaseError("Simulated DB failure")
    else:
        mock_conn = MagicMock()
        mock_conn.__bool__ = lambda self: connected  # makes `if conn:` work
        mock_db.get_connection.return_value = mock_conn
    return mock_db


def _make_advisor_mock(
    provider_status: dict | None = None,
    raise_error: bool = False,
) -> MagicMock:
    """
    Build a mock for the AdvisorService class.

    Args:
        provider_status: Dict to return from check_provider_status().
        raise_error:     If True raises a generic Exception instead.
    """
    mock_advisor = MagicMock()
    if raise_error:
        mock_advisor.check_provider_status.side_effect = Exception("AI API timeout")
    else:
        status = (
            provider_status
            if provider_status is not None
            else {"gemini": "active", "groq": "standby"}
        )
        mock_advisor.check_provider_status.return_value = status
    return mock_advisor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def healthy_response():
    """
    Fixture: issue GET /health with both DB and AI mocked as healthy.
    Returns the raw Response.
    """
    with patch("api.main.Database") as mock_db_cls, patch(
        "api.main.AdvisorService"
    ) as mock_advisor_cls:
        mock_db_cls.return_value = _make_db_mock(connected=True)
        mock_advisor_cls.return_value = _make_advisor_mock()

        response = client.get("/health")
        yield response


@pytest.fixture
def db_failure_response():
    """
    Fixture: issue GET /health with the DB raising DatabaseError.
    Returns the raw Response for both status-code and body assertions.
    """
    with patch("api.main.Database") as mock_db_cls, patch(
        "api.main.AdvisorService"
    ) as mock_advisor_cls:
        mock_db_cls.return_value = _make_db_mock(raise_error=True)
        mock_advisor_cls.return_value = _make_advisor_mock()

        response = client.get("/health")
        yield response


@pytest.fixture
def ai_failure_response():
    """
    Fixture: issue GET /health with the DB healthy but AI raising an Exception.
    Returns the raw Response for both status-code and body assertions.
    """
    with patch("api.main.Database") as mock_db_cls, patch(
        "api.main.AdvisorService"
    ) as mock_advisor_cls:
        mock_db_cls.return_value = _make_db_mock(connected=True)
        mock_advisor_cls.return_value = _make_advisor_mock(raise_error=True)

        response = client.get("/health")
        yield response


# ---------------------------------------------------------------------------
# ── HTTP Status Code Tests ──────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestHealthHttpStatus:
    """Tests that verify raw HTTP status codes from the health endpoint."""

    def test_health_returns_http_200(self, healthy_response):
        """GET /health must always return 200 OK when the API is running."""
        assert healthy_response.status_code == 200

    def test_health_still_200_when_db_fails(self, db_failure_response):
        """
        GET /health must return 200 even when the DB is down.
        A degraded state is communicated in the body, not via HTTP error codes,
        because the API itself is reachable.
        """
        assert db_failure_response.status_code == 200

    def test_health_still_200_when_ai_fails(self, ai_failure_response):
        """GET /health must return 200 even when AI provider check raises."""
        assert ai_failure_response.status_code == 200


# ---------------------------------------------------------------------------
# ── Response Schema Tests ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestHealthResponseSchema:
    """Tests that verify the structure and types of the /health response body."""

    REQUIRED_FIELDS = {"status", "version", "database_connection", "ai_provider_status"}

    def test_schema_has_all_required_fields(self, healthy_response):
        """All four required fields must be present in the response."""
        body = healthy_response.json()
        missing = self.REQUIRED_FIELDS - body.keys()
        assert not missing, f"Response is missing fields: {missing}"

    def test_status_field_is_a_string(self, healthy_response):
        """The 'status' field must be a string."""
        assert isinstance(healthy_response.json()["status"], str)

    def test_version_field_is_a_string(self, healthy_response):
        """The 'version' field must be a string."""
        assert isinstance(healthy_response.json()["version"], str)

    def test_database_connection_field_is_a_string(self, healthy_response):
        """The 'database_connection' field must be a string."""
        assert isinstance(healthy_response.json()["database_connection"], str)

    def test_ai_provider_status_field_is_a_dict(self, healthy_response):
        """The 'ai_provider_status' field must be a dict/object."""
        assert isinstance(healthy_response.json()["ai_provider_status"], dict)

    def test_no_extra_unexpected_fields(self, healthy_response):
        """
        The response must not contain undocumented fields.
        This guards against accidental data leaks (e.g. internal keys).
        """
        body = healthy_response.json()
        extra = body.keys() - self.REQUIRED_FIELDS
        assert not extra, f"Response has unexpected extra fields: {extra}"


# ---------------------------------------------------------------------------
# ── Business Logic / Happy-Path Tests ──────────────────────────────────────
# ---------------------------------------------------------------------------


class TestHealthHappyPath:
    """Tests that verify correct values in the response when everything is healthy."""

    def test_version_value_is_correct(self, healthy_response):
        """
        The 'version' field must exactly match the application version string.
        This acts as a regression guard against accidental version bumps.
        """
        assert healthy_response.json()["version"] == "1.0.0"

    def test_status_is_healthy_when_db_connected(self, healthy_response):
        """When the DB is reachable, 'status' must be 'healthy'."""
        assert healthy_response.json()["status"] == "healthy"

    def test_database_connection_is_connected(self, healthy_response):
        """When the DB is reachable, 'database_connection' must be 'connected'."""
        assert healthy_response.json()["database_connection"] == "connected"

    def test_ai_provider_status_is_forwarded(self, healthy_response):
        """
        The mock advisor returns {"gemini": "active", "groq": "standby"};
        the endpoint must forward this dict unchanged into the response.
        """
        expected = {"gemini": "active", "groq": "standby"}
        assert healthy_response.json()["ai_provider_status"] == expected


# ---------------------------------------------------------------------------
# ── Database Failure Path Tests ─────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestHealthDatabaseFailure:
    """Tests that verify correct degraded behaviour when the DB is unavailable."""

    def test_status_is_degraded_when_db_fails(self, db_failure_response):
        """When the DB raises DatabaseError, 'status' must be 'degraded'."""
        assert db_failure_response.json()["status"] == "degraded"

    def test_database_connection_is_disconnected_on_failure(self, db_failure_response):
        """When the DB raises DatabaseError, 'database_connection' must be 'disconnected'."""
        assert db_failure_response.json()["database_connection"] == "disconnected"

    def test_ai_provider_status_still_returned_on_db_failure(self, db_failure_response):
        """
        AI provider status must still be populated even when the DB fails,
        because the two checks are independent try/except blocks.
        """
        ai_status = db_failure_response.json()["ai_provider_status"]
        assert isinstance(ai_status, dict)
        assert len(ai_status) > 0


# ---------------------------------------------------------------------------
# ── AI Provider Failure Path Tests ─────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestHealthAiProviderFailure:
    """Tests that verify graceful handling when the AI provider check raises."""

    def test_status_remains_healthy_when_only_ai_fails(self, ai_failure_response):
        """
        If the DB is connected but AI check raises, the overall status must
        still be 'healthy' — AI failure alone does not degrade the API.
        """
        assert ai_failure_response.json()["status"] == "healthy"

    def test_database_remains_connected_when_ai_fails(self, ai_failure_response):
        """DB connection status must not be affected by an AI provider error."""
        assert ai_failure_response.json()["database_connection"] == "connected"

    def test_ai_error_is_captured_in_provider_status(self, ai_failure_response):
        """
        When the AI check raises, the exception message must be captured
        inside ai_provider_status under the key 'error'.
        """
        ai_status = ai_failure_response.json()["ai_provider_status"]
        assert "error" in ai_status

    def test_ai_error_message_is_correct(self, ai_failure_response):
        """
        The captured error string must match the exception message raised
        by the mock — verifying the error is forwarded verbatim.
        """
        ai_status = ai_failure_response.json()["ai_provider_status"]
        assert ai_status["error"] == "AI API timeout"


# ---------------------------------------------------------------------------
# ── Dual-Failure Edge Case ──────────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestHealthDualFailure:
    """Edge case: both DB and AI provider fail simultaneously."""

    def test_both_db_and_ai_fail_returns_200(self):
        """The endpoint must still return HTTP 200 even on dual failure."""
        with patch("api.main.Database") as mock_db_cls, patch(
            "api.main.AdvisorService"
        ) as mock_advisor_cls:
            mock_db_cls.return_value = _make_db_mock(raise_error=True)
            mock_advisor_cls.return_value = _make_advisor_mock(raise_error=True)

            response = client.get("/health")

        assert response.status_code == 200

    def test_both_db_and_ai_fail_status_is_degraded(self):
        """When both fail, overall 'status' must be 'degraded'."""
        with patch("api.main.Database") as mock_db_cls, patch(
            "api.main.AdvisorService"
        ) as mock_advisor_cls:
            mock_db_cls.return_value = _make_db_mock(raise_error=True)
            mock_advisor_cls.return_value = _make_advisor_mock(raise_error=True)

            response = client.get("/health")

        data = response.json()
        assert data["status"] == "degraded"
        assert data["database_connection"] == "disconnected"
        assert "error" in data["ai_provider_status"]


# ---------------------------------------------------------------------------
# ── HTTP Method Contract Tests ──────────────────────────────────────────────
# ---------------------------------------------------------------------------


class TestHealthHttpMethodContract:
    """
    The /health route is GET-only.
    Any other verb must be rejected with 405 Method Not Allowed.
    These tests guard against accidental route mis-configuration.
    """

    def test_post_to_health_returns_405(self):
        response = client.post("/health")
        assert response.status_code == 405

    def test_put_to_health_returns_405(self):
        response = client.put("/health")
        assert response.status_code == 405

    def test_delete_to_health_returns_405(self):
        response = client.delete("/health")
        assert response.status_code == 405

    def test_patch_to_health_returns_405(self):
        response = client.patch("/health")
        assert response.status_code == 405
