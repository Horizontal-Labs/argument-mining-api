from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str | None = None
    adu_classifier_model: str
    stance_classifier_model: str
    message: str
    # Back-compat global toggle; if provided and specific flags are None,
    # it applies to both ADU and stance.
    use_few_shot: bool | None = None
    # New: independent few-shot toggles
    use_few_shot_adu: bool | None = None
    use_few_shot_stance: bool | None = None

class ChatError(BaseModel):  
    detail: str
