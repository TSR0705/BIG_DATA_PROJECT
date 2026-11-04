"""Routes package."""

from api.app.routes.health import router as health_router
from api.app.routes.project import router as project_router
from api.app.routes.model import router as model_router
from api.app.routes.research import router as research_router
from api.app.routes.applicants import router as applicants_router
from api.app.routes.predictions import router as predictions_router

__all__ = [
    "health_router",
    "project_router",
    "model_router",
    "research_router",
    "applicants_router",
    "predictions_router",
]
