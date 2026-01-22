# app/core/hub_client.py
"""
Matrix-Hub client for matrix-ai.

IMPORTANT: matrix-ai is a read-only planning/reasoning service.
It should NEVER call admin/mutation endpoints on Matrix-Hub.
This client enforces that constraint.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from .config import Settings

logger = logging.getLogger(__name__)

# Endpoints that matrix-ai must NEVER call (admin/mutation operations)
FORBIDDEN_PATHS = ("/install", "/remotes", "/ingest", "/sync")


class MatrixHubClient:
    """
    Async HTTP client for Matrix-Hub with read-only enforcement.

    - Automatically includes Bearer token if configured
    - Fails fast if code attempts to call admin/mutation endpoints
    - Provides operator-friendly error messages on 401/403
    """

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or Settings.load()
        self.base_url = str(self._settings.matrixhub.base_url).rstrip("/")
        self.token = self._settings.matrixhub.token

    def _assert_read_only(self, path: str) -> None:
        """
        Guard to prevent accidental calls to admin/mutation endpoints.
        Raises RuntimeError if path contains forbidden segments.
        """
        if any(forbidden in path for forbidden in FORBIDDEN_PATHS):
            raise RuntimeError(
                f"matrix-ai must not call admin/mutation endpoints on Matrix-Hub. "
                f"Attempted path: {path}"
            )

    def _headers(self) -> Dict[str, str]:
        """Build request headers with optional Bearer token."""
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def get(self, path: str) -> Any:
        """
        Perform a GET request to Matrix-Hub.

        Args:
            path: API path (e.g., "/catalog/agents")

        Returns:
            Parsed JSON response

        Raises:
            RuntimeError: If path is a forbidden admin endpoint
            HTTPException: On HTTP errors with operator-friendly messages
        """
        self._assert_read_only(path)

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(
                f"{self.base_url}{path}",
                headers=self._headers(),
            )

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (401, 403):
                logger.warning(
                    "Matrix-Hub auth error %s on %s: %s",
                    resp.status_code, path, resp.text
                )
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=(
                        "Matrix-Hub authorization error. "
                        "This service should only use public/read-only endpoints. "
                        "If admin access is intended, set MATRIX_HUB_TOKEN."
                    ),
                )

            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Matrix-Hub error: {resp.text}",
            )

    async def post(self, path: str, data: Optional[Dict[str, Any]] = None) -> Any:
        """
        Perform a POST request to Matrix-Hub (for read-only query endpoints only).

        Args:
            path: API path
            data: JSON body to send

        Returns:
            Parsed JSON response

        Raises:
            RuntimeError: If path is a forbidden admin endpoint
            HTTPException: On HTTP errors with operator-friendly messages
        """
        self._assert_read_only(path)

        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{self.base_url}{path}",
                headers=self._headers(),
                json=data or {},
            )

            if resp.status_code in (200, 201):
                return resp.json()

            if resp.status_code in (401, 403):
                logger.warning(
                    "Matrix-Hub auth error %s on %s: %s",
                    resp.status_code, path, resp.text
                )
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=(
                        "Matrix-Hub authorization error. "
                        "This service should only use public/read-only endpoints. "
                        "If admin access is intended, set MATRIX_HUB_TOKEN."
                    ),
                )

            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Matrix-Hub error: {resp.text}",
            )


# Convenience factory
def get_hub_client(settings: Optional[Settings] = None) -> MatrixHubClient:
    """Create a MatrixHubClient instance."""
    return MatrixHubClient(settings)


__all__ = ["MatrixHubClient", "get_hub_client", "FORBIDDEN_PATHS"]
