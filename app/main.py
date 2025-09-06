from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import chat, health, models
import uvicorn
from app.log import log
import os
from app.argmining.config import OPENAI_KEY, HF_TOKEN, _dotenv_contains
from app.argmining.config import ENV_PATHS_TRIED
from pathlib import Path
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: log .env location and presence of critical env vars (no secrets)
    # Check runtime env (fresh) and presence in .env file
    oai_set = bool(os.getenv("OPEN_AI_KEY") or os.getenv("OPENAI_API_KEY") or OPENAI_KEY)
    hf_set = bool(os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN") or HF_TOKEN)
    # Nicely formatted environment diagnostics for all paths we try
    log().info("Config (.env) paths (checked in order):")
    for p in ENV_PATHS_TRIED:
        log().info(f"  {p}  -->  {'✅ found' if Path(p).exists() else '❌ not found'}")
    log().info(f"OPENAI_KEY:\n  {'✅ set' if oai_set else '❌ not set'}")
    in_env_file = _dotenv_contains("HF_TOKEN") or _dotenv_contains("HUGGINGFACEHUB_API_TOKEN")
    log().info(f"HF_TOKEN:\n  {'✅ set' if hf_set else '❌ not set'}")
    if not hf_set and in_env_file:
        log().warning("HF_TOKEN appears in .env but is not present in process env — check .env formatting or encoding; loading uses UTF-8 and override=True.")
    if not hf_set:
        log().warning("HF_TOKEN not set — some local HF models may be unavailable or fail to download.")
    if not oai_set:
        log().warning("OPENAI_KEY not set — OpenAI-backed steps (e.g., linking) will fail.")

    # Log registered routes for debugging missing endpoints
    try:
        from fastapi.routing import APIRoute
        log().info("Registered routes:")
        for route in app.router.routes:
            if isinstance(route, APIRoute):
                methods = ",".join(sorted(m for m in route.methods if m))
                log().info(f"  {methods:>10}  {route.path}")
    except Exception:
        pass

    yield
    # Shutdown: nothing special for now

def create_app() -> FastAPI:
    app = FastAPI(
        title="Argument Mining API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # include our routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(chat.router, prefix="/chat", tags=["chat"])
    app.include_router(models.router, prefix="/models", tags=["models"])

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
