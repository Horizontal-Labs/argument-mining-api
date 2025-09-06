import os
from pathlib import Path

from dotenv import load_dotenv, dotenv_values

# Candidate .env locations (ordered from nearest to broadest so later can override)
_ARGMINING_DIR = Path(__file__).parent
_APP_DIR = _ARGMINING_DIR.parent
_PROJECT_ROOT = _APP_DIR.parent

ENV_PATHS_TRIED = [
    _ARGMINING_DIR / ".env",
    _APP_DIR / ".env",
    _PROJECT_ROOT / ".env",
]

# Load all candidates (later files can override earlier ones)
for _p in ENV_PATHS_TRIED:
    try:
        load_dotenv(dotenv_path=_p, override=True, encoding="utf-8")
    except Exception:
        # Ignore load errors; presence will be checked separately
        pass

# Resolve keys (non-secret values only used elsewhere for presence checks)
OPENAI_KEY = os.getenv("OPEN_AI_KEY") or os.getenv("OPENAI_API_KEY")
# Support common HF token aliases
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Expose a helper for diagnostics
def _dotenv_contains(key: str) -> bool:
    try:
        for _p in ENV_PATHS_TRIED:
            vals = dotenv_values(_p)
            if key in vals and bool(vals.get(key)):
                return True
        return False
    except Exception:
        return False
