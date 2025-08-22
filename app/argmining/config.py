import os
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)
OPENAI_KEY = os.getenv("OPEN_AI_KEY") or os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")