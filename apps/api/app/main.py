from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import Base, engine
from app.routes.auth import router as auth_router
from app.routes.contracts import router as contracts_router
from app.routes.health import router as health_router
from app.routes.jobs import router as jobs_router

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)


app.include_router(health_router)
app.include_router(auth_router)
app.include_router(contracts_router)
app.include_router(jobs_router)

