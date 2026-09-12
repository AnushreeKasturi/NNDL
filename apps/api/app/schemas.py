from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: UUID
    email: EmailStr
    created_at: datetime

    model_config = {"from_attributes": True}


class ContractCreateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    text: str = Field(min_length=1)


class ContractOut(BaseModel):
    id: UUID
    title: str
    text: str
    created_at: datetime

    model_config = {"from_attributes": True}


class AnalyzeRequest(BaseModel):
    model_name: str | None = None


class JobOut(BaseModel):
    id: UUID
    owner_id: UUID
    contract_id: UUID
    model_name: str
    status: str
    result_json: dict[str, Any] | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

