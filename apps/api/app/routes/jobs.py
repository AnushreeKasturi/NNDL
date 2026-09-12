from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.deps import get_current_user
from app.models import InferenceJob, User
from app.schemas import JobOut

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobOut)
def get_job(
    job_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> InferenceJob:
    try:
        job_uuid = UUID(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid job id") from exc

    job = db.scalar(
        select(InferenceJob).where(InferenceJob.id == job_uuid, InferenceJob.owner_id == current_user.id)
    )
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


@router.get("", response_model=list[JobOut])
def list_jobs(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[InferenceJob]:
    return list(
        db.scalars(
            select(InferenceJob)
            .where(InferenceJob.owner_id == current_user.id)
            .order_by(InferenceJob.created_at.desc())
        )
    )
