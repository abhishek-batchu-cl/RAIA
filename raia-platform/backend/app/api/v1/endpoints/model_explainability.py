"""
RAIA Platform - Model Explainability Endpoints
"""

from fastapi import APIRouter

router = APIRouter()

# Placeholder for model explainability endpoints
@router.get("/")
async def list_explanations():
    return {"message": "Model explainability endpoints coming soon"}