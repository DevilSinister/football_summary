from pathlib import Path

import bcrypt
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select

from .database import Base, SessionLocal, engine
from .schema import ensure_schema
from .models import User
from .routers import processing, users


STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
Base.metadata.create_all(bind=engine)
# create_all never alters an existing table; add teams.short_code and the
# events clip-time columns if missing.
ensure_schema(engine)

app = FastAPI(
    title="Footy AI API",
    description="Flutter, SQL Server, and football video-processing backend",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(users.router)
app.include_router(processing.router)


@app.on_event("startup")
def delete_stale_uploads() -> None:
    # A restart empties the in-memory job table, so an upload left by a job
    # that was running is only reclaimed here, once it is old enough.
    # Highlights (clips, reels, player images) are never deleted.
    processing.sweep_media()


@app.on_event("startup")
def seed_demo_user() -> None:
    with SessionLocal() as db:
        if db.scalar(select(User).where(User.email == "demo@footyai.com")) is None:
            db.add(
                User(
                    username="demo_user",
                    email="demo@footyai.com",
                    password_hash=bcrypt.hashpw(b"password123", bcrypt.gensalt()).decode(),
                )
            )
            db.commit()


@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "healthy"}
