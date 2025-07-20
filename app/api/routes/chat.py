from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from app.api.schemas.chat import ChatRequest, ChatError
from app.api.services import preprocessor
from app.api.utils.session import ensure_session
from ...log import log
from app.api.services.model_client import run_argument_mining
import json


router = APIRouter()

ALLOWED_MODELS = {"modernbert", "openai", "tinyllama"}

@router.post(
    "/send",
    response_class=StreamingResponse,
    responses={
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
async def send_chat(payload: ChatRequest):
    if payload.model not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model not available"
        )

    if not payload.message or not payload.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message is required"
        )

    session_id = ensure_session(payload.session_id)
    model = payload.model
    cleaned = preprocessor.clean_text(payload.message)

    try:
        response = run_argument_mining(model, cleaned)
        log().info(f"Model response: {response}")
    except Exception as e:
        log().error(f"Model inference failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model processing failed: {str(e)}"
        )

    return StreamingResponse(
        json.dumps({
            "message": cleaned,
            "session_id": session_id,
            "model": model,
            "output": response  # returning model output to client
        }),
        media_type="application/json"
    )

#    # (c) call external model server
#    try:
#        ml_response = await model_client.render_diagram({
#            "session_id": session_id,
#            "text": cleaned
#        })
#    except Exception as e:
#        raise HTTPException(500, detail=f"Model service error: {e}")
#
#    # (d) stream binary image back
#    return StreamingResponse(
#        ml_response.aiter_bytes(),
#        media_type="image/png"
#    )
