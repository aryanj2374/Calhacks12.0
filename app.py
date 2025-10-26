"""FastAPI service exposing the dining chat assistant."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

from chat_agent import ChatResult, DiningChatAgent, DEFAULT_MODEL, load_agent

app = FastAPI(title="Cal Dining Assistant")

AGENT: Optional[DiningChatAgent] = None


class ChatRequest(BaseModel):
    query: str = Field(..., description="Natural language request from the user.")
    top_k: int = Field(5, ge=1, le=10, description="Maximum number of dishes to return.")
    user_id: Optional[str] = Field(
        None,
        description="Optional user identifier to personalize results.",
    )


class RecommendationItem(BaseModel):
    name: str
    item_id: str
    serving: Dict[str, Any]
    dietary: Dict[str, Any]
    nutrition: Dict[str, Any]
    blurb: str
    metadata: Dict[str, Any]
    preference_score: float
    classification_hints: Dict[str, Any]


class ChatResponse(BaseModel):
    used_llm: bool
    response: str
    items: list[RecommendationItem]
    memory_used: bool
    memory_context: Optional[Dict[str, Any]]


class FeedbackRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Identifier for the end user.")
    item_id: str = Field(..., min_length=1, description="Identifier for the menu item.")
    vote: Literal["upvote", "downvote"] = Field(
        ...,
        description="Whether the user liked (upvote) or disliked (downvote) the item.",
    )


@app.on_event("startup")
def startup() -> None:
    """Initialize the chat agent once when the API starts."""
    global AGENT
    menus_path = Path(os.getenv("MENUS_JSON", "menus.json"))
    if not menus_path.exists():
        raise RuntimeError(f"menus.json not found at {menus_path}. Run scraper.py first.")

    model = os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
    AGENT = load_agent(menus_path, model=model)


@app.get("/health", tags=["meta"])
def health() -> Dict[str, Any]:
    """Simple health check."""
    return {
        "status": "ok",
        "model": os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
        "menus_cached": bool(Path(os.getenv("MENUS_JSON", "menus.json")).exists()),
        "ollama_host": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    }


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest) -> ChatResponse:
    """Return grounded recommendations for the provided query."""
    if AGENT is None:
        raise HTTPException(status_code=503, detail="Assistant is still loading. Try again shortly.")

    result: ChatResult = AGENT.respond(
        request.query,
        top_k=request.top_k,
        user_id=request.user_id,
    )
    formatted = [_format_item_for_api(item) for item in result.recommendations]
    return ChatResponse(
        used_llm=result.used_llm,
        response=result.response,
        items=formatted,
        memory_used=result.memory_used,
        memory_context=result.memory_context,
    )


@app.post("/feedback", tags=["chat"])
def feedback(request: FeedbackRequest) -> Dict[str, Any]:
    """Capture user feedback (upvote or downvote) on a recommendation."""
    if AGENT is None:
        raise HTTPException(status_code=503, detail="Assistant is still loading. Try again shortly.")

    vote_value = 1 if request.vote == "upvote" else -1
    try:
        feedback_result = AGENT.record_feedback(request.user_id, request.item_id, vote_value)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    payload = {"status": "recorded"}
    payload.update(feedback_result)
    return payload


def _format_item_for_api(item: Dict[str, Any]) -> RecommendationItem:
    serving = {
        "location": item.get("location"),
        "meal": item.get("meal"),
        "hours": item.get("hours") or [],
        "hours_structured": item.get("hours_structured") or [],
        "status": item.get("status"),
        "crowdedness": item.get("crowdedness"),
    }
    dietary = {
        "choices": item.get("dietary_choices") or [],
        "tags": item.get("tags") or [],
    }
    metadata = {
        "category": item.get("category"),
        "score": item.get("score"),
        "menu_reference": item.get("menu_reference"),
    }
    return RecommendationItem(
        name=item.get("name", "Unknown Item"),
        item_id=item.get("item_id", ""),
        serving=serving,
        dietary=dietary,
        nutrition=item.get("nutrition") or {},
        blurb=item.get("blurb") or "",
        metadata=metadata,
        preference_score=float(item.get("preference_score") or 0.0),
        classification_hints=item.get("classification_hints") or {},
    )
