from fastapi import APIRouter, Request

from src.api_secrets import clear_api_key, set_api_key
from src.dependencies import limiter
from src.schemas import ApiKeyRequest

router = APIRouter()


@router.post("/config/api-key", status_code=204)
@limiter.limit("5/minute")
async def config_set_api_key(request: Request, body: ApiKeyRequest):
    set_api_key(body.api_key)


@router.delete("/config/api-key", status_code=204)
async def config_clear_api_key():
    clear_api_key()
