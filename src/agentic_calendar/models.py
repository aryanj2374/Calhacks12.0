from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class EventCategory(str, Enum):
    LECTURE = "lecture"
    DISCUSSION = "discussion"
    LAB = "lab"
    MIDTERM = "midterm"
    FINAL = "final"
    REVIEW = "review"
    HOMEWORK = "homework"
    PROJECT = "project"
    QUIZ = "quiz"
    OTHER = "other"


class CourseEvent(BaseModel):
    """Structured representation of a schedule entry."""

    title: str
    category: EventCategory = EventCategory.OTHER
    start: datetime
    end: Optional[datetime] = None
    location: Optional[str] = None
    description: Optional[str] = None
    source_url: Optional[str] = None

    @validator("end")
    def validate_order(cls, end: Optional[datetime], values: dict[str, object]) -> Optional[datetime]:  # noqa: D417
        start = values.get("start")
        if end is not None and start is not None and end < start:
            raise ValueError("event end must be after start")
        return end


class ExtractionReport(BaseModel):
    source_url: str
    events: List[CourseEvent] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    raw_blocks: List[str] = Field(default_factory=list)

    def summary(self) -> str:
        kind_counts: dict[EventCategory, int] = {}
        for event in self.events:
            kind_counts[event.category] = kind_counts.get(event.category, 0) + 1
        breakdown = ", ".join(f"{cat.name.title()}: {count}" for cat, count in kind_counts.items())
        return f"{len(self.events)} events extracted; {breakdown or 'no breakdown available'}"
