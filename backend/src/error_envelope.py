"""Centralised error-response envelope for the API.

Every error response has the shape::

    {"error": {"code": "<SCREAMING_SNAKE>", "message": "<human string>"}}

Register the four handlers in create_app so FastAPI/Starlette errors are also
wrapped automatically.  The global_exception_handler is the last-resort
fallback: it catches any unhandled Exception, logs the full traceback
server-side, and returns a generic 500 so internal details never reach clients.
"""

from dataclasses import dataclass

import structlog
from fastapi import Request
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse

logger = structlog.get_logger(__name__)


@dataclass
class AppError(Exception):
    """Raise instead of HTTPException to produce the standardised envelope."""

    status_code: int
    code: str
    message: str


def _body(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_body(exc.code, exc.message),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    _fallback_codes: dict[int, str] = {
        400: "BAD_REQUEST",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        500: "INTERNAL_ERROR",
        502: "BAD_GATEWAY",
    }
    code = _fallback_codes.get(exc.status_code, f"HTTP_{exc.status_code}")
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content=_body(code, message),
    )


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=_body("VALIDATION_ERROR", str(exc.errors())),
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler: log the full traceback, return a generic 500."""
    logger.error(
        "unhandled_exception",
        path=str(request.url.path),
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(
        status_code=500,
        content=_body("INTERNAL_ERROR", "An unexpected error occurred."),
    )
