"""FastAPI application entry point."""
import os
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.core.database import init_db
from app.core.graph_db import init_graph_schema, close_driver
from app.core.redis_client import close_redis
from app.api import auth, chat, knowledge

setup_logging()
settings = get_settings()
logger = get_logger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Enterprise Brain", env=settings.app_env)
    await init_db()
    try:
        await init_graph_schema()
    except Exception as e:
        logger.warning("Neo4j schema init skipped", error=str(e))

    # Set LangSmith env vars
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

    yield

    # Shutdown
    await close_driver()
    await close_redis()
    logger.info("Enterprise Brain stopped")


app = FastAPI(
    title="Enterprise Brain API",
    description="企业智脑 — 企业级 AI 知识助手",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        "HTTP",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        ms=round(elapsed, 1),
    )
    return response


# Routers
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(knowledge.router)


@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok", "service": "enterprise-brain"}
