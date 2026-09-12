from __future__ import annotations

from sqlalchemy import select

from app.config import settings
from app.db import SessionLocal
from app.models import Contract, InferenceJob
from app.services.classifier import infer_document


def run_inference_job(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.scalar(select(InferenceJob).where(InferenceJob.id == job_id))
        if job is None:
            raise ValueError(f"Job not found: {job_id}")

        contract = db.scalar(select(Contract).where(Contract.id == job.contract_id))
        if contract is None:
            job.status = "failed"
            job.error_message = "Contract not found"
            db.commit()
            return

        job.status = "running"
        job.error_message = None
        db.commit()

        result = infer_document(
            text=contract.text,
            model_name=job.model_name,
            max_length=settings.infer_max_length,
            stride=settings.infer_stride,
            threshold=settings.infer_threshold,
        )
        job.status = "succeeded"
        job.result_json = result
        db.commit()
    except Exception as exc:
        failed_job = db.scalar(select(InferenceJob).where(InferenceJob.id == job_id))
        if failed_job is not None:
            failed_job.status = "failed"
            failed_job.error_message = str(exc)
            db.commit()
        raise
    finally:
        db.close()

