"""
HTTP layer only -- every route delegates to app/services/*. Matches the
documented API contract: the 5 original endpoints plus /api/metrics for
the threshold-calibration requirement.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db import get_db
from app.schemas import (
    HealthOut,
    IncidentListOut,
    IncidentOut,
    IncidentPatch,
    MetricsOut,
    SummaryOut,
    ThresholdPatch,
)
from app.schemas_predict import PredictIn, PredictOut
from app.services import incidents_service, metrics_service, predict_service, summary_service

router = APIRouter(prefix="/api")


@router.get("/health", response_model=HealthOut)
def health():
    return HealthOut()


@router.get("/incidents", response_model=IncidentListOut)
def get_incidents(
    severity: list[str] | None = Query(default=None),
    attack_type: str | None = None,
    status: str | None = None,
    q: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    total, items = incidents_service.list_incidents(
        db, severity=severity, attack_type=attack_type, status=status, q=q,
        limit=limit, offset=offset,
    )
    # list endpoint intentionally omits explanation to stay light
    out_items = []
    for inc in items:
        out = IncidentOut.model_validate(inc)
        out.explanation = None
        out_items.append(out)
    return IncidentListOut(total=total, items=out_items)


@router.get("/incidents/{incident_id}", response_model=IncidentOut)
def get_incident(incident_id: str, db: Session = Depends(get_db)):
    inc = incidents_service.get_incident(db, incident_id)
    if inc is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return inc


@router.patch("/incidents/{incident_id}", response_model=IncidentOut)
def patch_incident(incident_id: str, body: IncidentPatch, db: Session = Depends(get_db)):
    inc = incidents_service.update_status(db, incident_id, body.status)
    if inc is None:
        raise HTTPException(status_code=404, detail="Incident not found")
    return inc


@router.get("/summary", response_model=SummaryOut)
def get_summary(db: Session = Depends(get_db)):
    return summary_service.get_summary(db)


@router.post("/predict", response_model=PredictOut)
def predict(body: PredictIn):
    """Score a single pasted CICIDS row and return the prediction + SHAP
    explanation. Stateless -- does not create an incident."""
    try:
        return predict_service.predict(body.row)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/metrics", response_model=MetricsOut)
def get_metrics(db: Session = Depends(get_db)):
    return metrics_service.get_current_metrics(db)


@router.patch("/metrics/threshold", response_model=MetricsOut)
def patch_threshold(body: ThresholdPatch, db: Session = Depends(get_db)):
    from app.config import get_settings

    settings = get_settings()
    settings.decision_threshold = body.threshold
    return metrics_service.get_current_metrics(db)