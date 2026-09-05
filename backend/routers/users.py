import bcrypt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User, UserLog
from ..schemas import AuthResponse, LoginRequest, RegisterRequest, UserResponse


router = APIRouter(prefix="/api/users", tags=["users"])


def _user_response(user: User) -> UserResponse:
    return UserResponse(userId=user.user_id, username=user.username, email=user.email)


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, request: Request, db: Session = Depends(get_db)):
    email = payload.email.lower().strip()
    username = payload.username.strip()
    duplicate = db.scalar(
        select(User).where(
            (func.lower(User.email) == email) | (func.lower(User.username) == username.lower())
        )
    )
    if duplicate:
        raise HTTPException(status_code=409, detail="Email or username is already registered.")

    password_hash = bcrypt.hashpw(payload.password.encode("utf-8"), bcrypt.gensalt()).decode()
    user = User(username=username, email=email, password_hash=password_hash)
    db.add(user)
    db.flush()
    db.add(
        UserLog(
            user_id=user.user_id,
            action="register",
            ip_address=request.client.host if request.client else None,
        )
    )
    db.commit()
    db.refresh(user)
    return AuthResponse(success=True, message="Account created successfully.", user=_user_response(user))


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(func.lower(User.email) == payload.email.lower().strip()))
    valid = user and bcrypt.checkpw(payload.password.encode("utf-8"), user.password_hash.encode())
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    db.add(
        UserLog(
            user_id=user.user_id,
            action="login",
            ip_address=request.client.host if request.client else None,
        )
    )
    db.commit()
    return AuthResponse(success=True, message="Login successful.", user=_user_response(user))


@router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    return _user_response(user)
