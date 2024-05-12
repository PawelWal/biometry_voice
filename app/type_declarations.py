"""Declarations of used types."""
from pydantic import BaseModel

class BiomUser(BaseModel):
    """User in system."""

    user_dir: str


class BiomVerify(BaseModel):
    """User to verify."""

    user_file : str
    user_cls : int


class BiomIdentify(BaseModel):
    """User to identify."""

    user_file : str
