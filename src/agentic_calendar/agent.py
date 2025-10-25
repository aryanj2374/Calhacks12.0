from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional

from dateutil import parser as date_parser
from pydantic import BaseModel, ValidationError, field_validator

from .calendar_client import GoogleCalendarClient
from .llm_client import LavaPaymentsLLMClient
from .models import CourseEvent, EventCategory
from .rag import CalendarRAGIndex


class CalendarAction(str, Enum):
    ADD_EVENT = "add_event"
    DELETE_EVENT = "delete_event"
    MOVE_EVENT = "move_event"
    LIST_EVENTS = "list_events"


@dataclass
class AgentResult:
    """Structured response sent back to the CLI/UI."""

    raw_response: str
    action: CalendarAction
    detail: str
    executed: bool


class CalendarAgent:
    """
    Coordinates RAG context, LavaPayments LLM interpretation, and Calendar execution.
    """

    SYSTEM_PROMPT = """
You are an assistant that translates natural-language calendar requests into JSON commands.
Always reply with valid JSON only, no extra prose. Use this schema:
{
  "action": "add_event | delete_event | move_event | list_events",
  "event": {
    "title": "string",
    "category": "lecture|discussion|lab|midterm|final|review|homework|project|quiz|other",
    "start": "ISO 8601 datetime",
    "end": "ISO 8601 datetime or null",
    "location": "string or null",
    "description": "string or null"
  },
  "target_title": "string describing the event to delete/move",
  "target_start": "ISO datetime narrowing the target event",
  "new_start": "ISO datetime for move operations",
  "new_end": "ISO datetime for move operations"
}
Rules:
- add_event requires an `event` block (title + start time).
- delete_event and move_event must include enough detail to uniquely identify an existing item.
- list_events just echoes the supplied calendar context (no mutation).
- Always resolve words like "today", "tomorrow", or "next week" using the current datetime string provided alongside the calendar context.
"""

    def __init__(
        self,
        *,
        events: List[CourseEvent],
        calendar_client: GoogleCalendarClient | None,
        llm_client: LavaPaymentsLLMClient,
    ):
        self.events = list(events)
        self.calendar_client = calendar_client
        self.llm_client = llm_client
        self._history: List[dict[str, str]] = []
        self._rag_index = CalendarRAGIndex(self.events)

    def handle_request(self, user_text: str, *, dry_run: bool = False) -> AgentResult:
        """Send the request to the LLM, parse the JSON command, and execute it."""

        context = self._rag_index.build_context(user_text)
        now_context = self._current_time_context()
        prompt = f"{now_context}\n\nCalendar context:\n{context}\n\nUser request:\n{user_text}"
        messages = [*self._history, {"role": "user", "content": prompt}]

        raw_response = self.llm_client.chat(messages, system_prompt=self.SYSTEM_PROMPT.strip())
        self._history.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": raw_response},
            ]
        )

        command = self._parse_command(raw_response)
        detail = self._execute_command(command, dry_run=dry_run)
        return AgentResult(
            raw_response=raw_response,
            action=command.action,
            detail=detail,
            executed=not dry_run,
        )

    def _execute_command(self, command: "CalendarCommand", *, dry_run: bool) -> str:
        if command.action == CalendarAction.LIST_EVENTS:
            focus = command.target_title or ""
            return self._rag_index.build_context(focus)

        if command.action == CalendarAction.ADD_EVENT:
            if command.event is None:
                raise ValueError("LLM response missing 'event' payload for add_event")
            course_event = command.event.to_course_event()
            self.events.append(course_event)
            self._rag_index = CalendarRAGIndex(self.events)
            start_iso = course_event.start.isoformat()
            end_iso = (course_event.end or course_event.start).isoformat()
            if dry_run or self.calendar_client is None:
                return (
                    f"[dry-run] Would create '{course_event.title}' "
                    f"{start_iso} → {end_iso}"
                )
            created = self.calendar_client.create_events([course_event])
            return (
                f"Created calendar event '{course_event.title}' "
                f"{start_iso} → {end_iso} (ids: {created})"
            )

        if command.action == CalendarAction.DELETE_EVENT:
            target = self._locate_event(command.target_title, command.target_start)
            self._remove_local_event(target)
            if dry_run or self.calendar_client is None:
                return f"[dry-run] Would delete '{target.title}' at {target.start.isoformat()}"
            deleted = self.calendar_client.delete_matching_events([target])
            return f"Deleted {deleted} events matching '{target.title}'"

        if command.action == CalendarAction.MOVE_EVENT:
            target = self._locate_event(command.target_title, command.target_start)
            if command.new_start is None:
                raise ValueError("move_event requires 'new_start'")
            new_start = _parse_datetime(command.new_start)
            new_end = _parse_datetime(command.new_end) if command.new_end else target.end

            updated = target.model_copy(update={"start": new_start, "end": new_end})
            self._remove_local_event(target, rebuild=False)
            self.events.append(updated)
            self._rag_index = CalendarRAGIndex(self.events)

            if dry_run or self.calendar_client is None:
                return (
                    f"[dry-run] Would move '{target.title}' from "
                    f"{target.start.isoformat()} to {updated.start.isoformat()}"
                )

            self.calendar_client.delete_matching_events([target])
            self.calendar_client.create_events([updated])
            return (
                f"Moved '{target.title}' from {target.start.isoformat()} "
                f"to {updated.start.isoformat()}"
            )

        raise ValueError(f"Unsupported action '{command.action}'")

    def _locate_event(self, title: Optional[str], start_iso: Optional[str]) -> CourseEvent:
        if not title:
            raise ValueError("LLM response missing target_title for delete/move operation")
        matches = [event for event in self.events if title.lower() in event.title.lower()]
        if start_iso:
            desired = _parse_datetime(start_iso)
            matches = [
                event for event in matches if abs((event.start - desired).total_seconds()) <= 60
            ]
        if not matches:
            raise ValueError(f"Could not find event matching '{title}'")
        return matches[0]

    def _remove_local_event(self, target: CourseEvent, *, rebuild: bool = True) -> None:
        self.events = [
            event
            for event in self.events
            if not (
                event.title == target.title
                and abs((event.start - target.start).total_seconds()) <= 60
            )
        ]
        if rebuild:
            self._rag_index = CalendarRAGIndex(self.events)

    @staticmethod
    def _parse_command(text: str) -> "CalendarCommand":
        try:
            json_payload = _extract_json(text)
            return CalendarCommand.model_validate_json(json_payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(f"LLM response was not valid command JSON: {text}") from exc

    def _current_time_context(self) -> str:
        tzinfo = self._infer_timezone()
        now = datetime.now(tz=tzinfo)
        readable = now.strftime("%A, %B %d %Y %H:%M %Z")
        return f"Current datetime: {now.isoformat()} ({readable})"

    def _infer_timezone(self):
        for event in self.events:
            if event.start.tzinfo:
                return event.start.tzinfo
        return timezone.utc


class EventPayload(BaseModel):
    title: str
    start: str
    end: Optional[str] = None
    category: Optional[str | EventCategory] = None
    location: Optional[str] = None
    description: Optional[str] = None

    @field_validator("category")
    @classmethod
    def _coerce_category(cls, value: Optional[str | EventCategory]) -> EventCategory | None:
        if value is None:
            return None
        if isinstance(value, EventCategory):
            return value
        try:
            return EventCategory(value.lower())
        except ValueError:
            return EventCategory.OTHER

    def to_course_event(self) -> CourseEvent:
        return CourseEvent(
            title=self.title,
            category=self.category or EventCategory.OTHER,
            start=date_parser.isoparse(self.start),
            end=date_parser.isoparse(self.end) if self.end else None,
            location=self.location,
            description=self.description,
            source_url="agentic_calendar",
        )


class CalendarCommand(BaseModel):
    action: CalendarAction
    event: Optional[EventPayload] = None
    target_title: Optional[str] = None
    target_start: Optional[str] = None
    new_start: Optional[str] = None
    new_end: Optional[str] = None


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("Missing JSON braces", text, 0)
    return text[start : end + 1]


def _parse_datetime(value: str) -> datetime:
    return date_parser.isoparse(value)
