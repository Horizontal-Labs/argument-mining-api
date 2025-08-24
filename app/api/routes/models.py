from fastapi import APIRouter
from typing import Dict, List

router = APIRouter()

# Define available models with their descriptions
AVAILABLE_MODELS = {
    "adu_classification": [
        {
            "id": "modernbert",
            "name": "ModernBERT",
            "description": "Fast and accurate BERT-based model with PEFT adapters",
            "provider": "local"
        },
        {
            "id": "deberta",
            "name": "DeBERTa",
            "description": "Lightweight encoder model for sentence classification",
            "provider": "local"
        },
        {
            "id": "tinyllama",
            "name": "TinyLlama",
            "description": "1.1B parameter language model optimized for CPU",
            "provider": "local"
        },
        {
            "id": "gpt-4.1",
            "name": "GPT-4.1",
            "description": "OpenAI's GPT-4.1 model - highest accuracy",
            "provider": "openai"
        },
        {
            "id": "gpt-5",
            "name": "GPT-5",
            "description": "OpenAI's GPT-5 model - next generation",
            "provider": "openai"
        },
        {
            "id": "gpt-5-mini",
            "name": "GPT-5 Mini",
            "description": "OpenAI's GPT-5 Mini - faster and more cost-effective",
            "provider": "openai"
        }
    ],
    "stance_classification": [
        {
            "id": "modernbert",
            "name": "ModernBERT",
            "description": "Fast and accurate BERT-based model with PEFT adapters",
            "provider": "local"
        },
        {
            "id": "deberta",
            "name": "DeBERTa",
            "description": "Lightweight encoder model for stance classification",
            "provider": "local"
        },
        {
            "id": "tinyllama",
            "name": "TinyLlama",
            "description": "1.1B parameter language model optimized for CPU",
            "provider": "local"
        },
        {
            "id": "gpt-4.1",
            "name": "GPT-4.1",
            "description": "OpenAI's GPT-4.1 model - highest accuracy",
            "provider": "openai"
        },
        {
            "id": "gpt-5",
            "name": "GPT-5",
            "description": "OpenAI's GPT-5 model - next generation",
            "provider": "openai"
        },
        {
            "id": "gpt-5-mini",
            "name": "GPT-5 Mini",
            "description": "OpenAI's GPT-5 Mini - faster and more cost-effective",
            "provider": "openai"
        }
    ]
}

@router.get("/available")
async def get_available_models() -> Dict[str, List[dict]]:
    """
    Get list of available models for ADU and stance classification.
    
    Returns:
        Dictionary with two keys:
        - adu_classification: List of available ADU classification models
        - stance_classification: List of available stance classification models
    """
    return AVAILABLE_MODELS

@router.get("/adu")
async def get_adu_models() -> List[dict]:
    """
    Get list of available ADU classification models.
    
    Returns:
        List of available ADU classification models with metadata
    """
    return AVAILABLE_MODELS["adu_classification"]

@router.get("/stance")
async def get_stance_models() -> List[dict]:
    """
    Get list of available stance classification models.
    
    Returns:
        List of available stance classification models with metadata
    """
    return AVAILABLE_MODELS["stance_classification"]