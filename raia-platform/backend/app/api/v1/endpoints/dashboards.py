"""
RAIA Platform - Dashboard Endpoints
"""

from fastapi import APIRouter

router = APIRouter()

# Placeholder for dashboard endpoints
@router.get("/")
async def get_dashboards():
    return {"message": "Dashboard endpoints coming soon"}