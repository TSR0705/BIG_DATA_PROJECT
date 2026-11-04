"""CreditLens FastAPI Backend Application."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.app.core.config import (
    ALLOWED_ORIGINS,
    DESCRIPTION,
    PROJECT_NAME,
    VERSION,
)
from api.app.routes import (
    applicants_router,
    health_router,
    model_router,
    predictions_router,
    project_router,
    research_router,
)
from api.app.services.model_service import get_inference_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CreditLensAPI")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload models on application startup."""
    logger.info("Starting CreditLens API...")
    try:
        engine = get_inference_engine()
        logger.info(f"Loaded ML Engine with {len(engine.feature_names)} features successfully.")
    except Exception as e:
        logger.error(f"Failed to preload model engine: {e}")
    yield
    logger.info("Shutting down CreditLens API...")


app = FastAPI(
    title=PROJECT_NAME,
    version=VERSION,
    description=DESCRIPTION,
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(health_router)
app.include_router(project_router)
app.include_router(model_router)
app.include_router(research_router)
app.include_router(applicants_router)
app.include_router(predictions_router)


@app.get("/")
def root():
    return {
        "project": "CreditLens - Loan Default Prediction",
        "status": "online",
        "docs_url": "/docs",
        "api_health": "/api/health",
        "api_overview": "/api/project/overview",
        "api_recent_predictions": "/api/predictions/recent",
    }
