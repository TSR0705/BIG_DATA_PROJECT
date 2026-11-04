"""Project overview router."""

from fastapi import APIRouter, HTTPException, Query

from api.app.schemas.project import DefaultRateProfile, ProjectOverview
from api.app.services import insights_service
from api.app.services.data_service import get_project_overview

router = APIRouter(prefix="/api/project", tags=["Project"])


@router.get("/overview", response_model=ProjectOverview)
def get_overview(profile: str = Query("age", description="Default-rate profile dimension: age | credit | income")) -> ProjectOverview:
    """Return project KPIs, applicant distribution, default-rate profile, SHAP risk factors and recent predictions.

    Every value is read from a committed project artifact or produced by the frozen model.
    """
    if profile not in insights_service.PROFILE_DIMENSIONS:
        raise HTTPException(status_code=422, detail=f"Unknown profile dimension '{profile}'")
    return get_project_overview(profile_dimension=profile)


@router.get("/default-rate-profile", response_model=DefaultRateProfile)
def get_default_rate_profile(dimension: str = Query("age")) -> DefaultRateProfile:
    """Observed default rate of the labelled population by applicant band (age, credit or income)."""
    if dimension not in insights_service.PROFILE_DIMENSIONS:
        raise HTTPException(status_code=422, detail=f"Unknown profile dimension '{dimension}'")
    return DefaultRateProfile(**insights_service.default_rate_profile(dimension))
