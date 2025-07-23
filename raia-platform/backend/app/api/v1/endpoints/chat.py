"""
RAIA Platform - Chat Endpoints
"""

from fastapi import APIRouter

router = APIRouter()

# Placeholder for chat endpoints
@router.get("/")
async def get_chat():
    return {"message": "Chat endpoints coming soon"}