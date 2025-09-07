from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from app.api.schemas.chat import ChatRequest, ChatError
from app.api.services import preprocessor
from app.api.utils.session import ensure_session
from app.log import logger
from app.api.services.model_client import run_argument_mining
import json


router = APIRouter()

ALLOWED_MODELS = {"modernbert", "tinyllama-finetuned", "tinyllama-base", "qwen2.5-1.5b", "deberta", "gpt-4.1", "gpt-5", "gpt-5-mini"}

@router.post(
    "/send",
    response_class=StreamingResponse,
    responses={
        400: {"description": "Bad Request"},
        500: {"description": "Internal Server Error"},
    },
)
async def send_chat(payload: ChatRequest):
    if payload.adu_classifier_model not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"ADU classifier model '{payload.adu_classifier_model}' not available"
        )
    
    if payload.stance_classifier_model not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stance classifier model '{payload.stance_classifier_model}' not available"
        )

    if not payload.message or not payload.message.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message is required"
        )

    session_id = ensure_session(payload.session_id)
    adu_model = payload.adu_classifier_model
    stance_model = payload.stance_classifier_model
    cleaned = preprocessor.clean_text(payload.message)
    # Few-shot toggles: specific flags override global flag
    global_fs = payload.use_few_shot
    use_few_shot_adu = payload.use_few_shot_adu if payload.use_few_shot_adu is not None else global_fs
    use_few_shot_stance = payload.use_few_shot_stance if payload.use_few_shot_stance is not None else global_fs
    use_few_shot_adu = bool(use_few_shot_adu) if use_few_shot_adu is not None else False
    use_few_shot_stance = bool(use_few_shot_stance) if use_few_shot_stance is not None else False

    try:
        response = run_argument_mining(
            adu_model,
            stance_model,
            cleaned,
            use_few_shot_adu,
            use_few_shot_stance,
        )
        logger.info(f"Model response: {response}")
    except Exception as e:
        logger.error(f"Model inference failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model processing failed: {str(e)}"
        )

    return StreamingResponse(
        json.dumps({
            "message": cleaned,
            "session_id": session_id,
            "adu_classifier_model": adu_model,
            "stance_classifier_model": stance_model,
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
