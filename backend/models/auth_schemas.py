from enum import Enum
from typing import List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

class UserRoleEnum(str, Enum):
    PUBLIC_USER = "PUBLIC_USER"
    HOSPITAL_DISPATCH = "HOSPITAL_DISPATCH"
    GOVERNMENT_OFFICIAL = "GOVERNMENT_OFFICIAL"

class LoginRequest(BaseModel):
    role: UserRoleEnum
    username: str
    password: str
    organization_name: Optional[str] = None

class LoginResponse(BaseModel):
    token: str
    role: UserRoleEnum
    username: str
    organization_name: str
    permissions: List[str]
    authenticated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserProfile(BaseModel):
    role: UserRoleEnum
    title: str
    description: str
    is_protected: bool
    allowed_actions: List[str]
