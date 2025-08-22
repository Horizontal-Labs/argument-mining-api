from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    adu_classifier_model: str
    stance_classifier_model: str
    message: str

class ChatError(BaseModel):  
    detail: str
