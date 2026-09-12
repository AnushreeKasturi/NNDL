from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import get_db
from app.deps import get_current_user
from app.models import Contract, InferenceJob, User
from app.schemas import AnalyzeRequest, ContractCreateRequest, ContractOut, JobOut
from app.services.queue import get_queue

router = APIRouter(prefix="/contracts", tags=["contracts"])


@router.post("", response_model=ContractOut, status_code=status.HTTP_201_CREATED)
def create_contract(
    payload: ContractCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Contract:
    contract = Contract(owner_id=current_user.id, title=payload.title, text=payload.text)
    db.add(contract)
    db.commit()
    db.refresh(contract)
    return contract


@router.get("", response_model=list[ContractOut])
def list_contracts(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> list[Contract]:
    return list(
        db.scalars(
            select(Contract).where(Contract.owner_id == current_user.id).order_by(Contract.created_at.desc())
        )
    )


@router.post("/{contract_id}/analyze", response_model=JobOut, status_code=status.HTTP_202_ACCEPTED)
def analyze_contract(
    contract_id: str,
    payload: AnalyzeRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> InferenceJob:
    try:
        contract_uuid = UUID(contract_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid contract id") from exc

    contract = db.scalar(
        select(Contract).where(Contract.id == contract_uuid, Contract.owner_id == current_user.id)
    )
    if contract is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Contract not found")

    model_name = payload.model_name or settings.default_model
    job = InferenceJob(
        owner_id=current_user.id,
        contract_id=contract.id,
        model_name=model_name,
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    queue = get_queue()
    queue.enqueue("app.tasks.run_inference_job", str(job.id))
    return job
