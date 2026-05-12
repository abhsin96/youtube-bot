from fastapi import APIRouter, Depends

from src.api_secrets import get_openai_key
from src.config import Settings
from src.dependencies import get_settings
from src.schemas import ConfigStatusResponse

router = APIRouter()


@router.get("/health")
async def health(settings: Settings = Depends(get_settings)):
    return {
        "status": "ok",
        "version": settings.version,
        "langsmith_enabled": settings.langsmith_tracing.lower() == "true",
        "has_api_key": bool(get_openai_key(settings)),
    }


@router.get("/config/status", response_model=ConfigStatusResponse)
async def config_status(settings: Settings = Depends(get_settings)):
    return ConfigStatusResponse(has_key=bool(get_openai_key(settings)))
