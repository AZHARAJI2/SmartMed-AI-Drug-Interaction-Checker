"""API package: FastAPI routers (patient, doctor_tool, scan) + schemas + app factory."""
from api.app import create_app

__all__ = ["create_app"]