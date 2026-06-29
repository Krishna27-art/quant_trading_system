import os
from datetime import datetime, timedelta

import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

router = APIRouter()
security = HTTPBearer()

ALGORITHM = "HS256"


def _jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY")
    if not secret or secret == "super-secret-institutional-key":
        raise RuntimeError("JWT_SECRET_KEY must be set to a non-default secret")
    return secret


def validate_auth_config() -> None:
    admin_password = os.getenv("ADMIN_PASSWORD")
    if not admin_password or admin_password == "admin":
        raise RuntimeError("ADMIN_PASSWORD must be set and cannot be 'admin'")
    _jwt_secret()


def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=1)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, _jwt_secret(), algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, _jwt_secret(), algorithms=[ALGORITHM])
        return payload
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


from fastapi.security import OAuth2PasswordRequestForm


@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate with credentials supplied through form data, preventing query param leakage.
    """
    try:
        validate_auth_config()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    expected_user = os.getenv("ADMIN_USERNAME", "admin")
    expected_password = os.getenv("ADMIN_PASSWORD")
    if form_data.username == expected_user and form_data.password == expected_password:
        access_token = create_access_token(data={"sub": form_data.username})
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(status_code=401, detail="Invalid credentials")
