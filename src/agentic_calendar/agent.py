from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo, timedelta
from enum import Enum
from typing import List, Optional, Sequence

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
- When the user gives a clock time without am/pm, interpret it as the next occurrence *after* the current time (e.g., if it's 5pm now and they say "7", schedule 7pm). Only fall back to the morning hour if the time has already passed for the day and the user clearly meant the next day.
- Apply the same rule to time ranges like "10-11" or "10 to 11": treat them as hour windows in the upcoming evening when the current time is later in the day, unless the user specifies otherwise.
- If the user explicitly says "today" with an ambiguous time (e.g., "today at 1130"), convert it to the evening slot when the current time is already in the afternoon/evening. Example: if the provided current datetime is 2025-02-01T20:46:00-08:00 and the user says "meeting today at 1130", you must schedule it for 2025-02-01T23:30:00-08:00 (11:30 PM), not 11:30 AM.
"""

    def __init__(
        self,
        *,
        events: List[CourseEvent],
        calendar_client: GoogleCalendarClient | None,
        llm_client: LavaPaymentsLLMClient,
        default_timezone: tzinfo | None = None,
    ):
        self.events = list(events)
        self.calendar_client = calendar_client
        self.llm_client = llm_client
        self._history: List[dict[str, str]] = []
        self._rag_index = CalendarRAGIndex(self.events)
        self._default_timezone = default_timezone

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
            executed = not dry_run and self.calendar_client is not None
            if executed:
                self.calendar_client.create_events([course_event])
            return self._format_add_message(course_event, executed=executed)

        if command.action == CalendarAction.DELETE_EVENT:
            target = self._locate_event(command.target_title, command.target_start)
            self._remove_local_event(target)
            executed = not dry_run and self.calendar_client is not None
            if executed:
                self.calendar_client.delete_matching_events([target])
            return self._format_delete_message(target, executed=executed)

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

            executed = not dry_run and self.calendar_client is not None
            if executed:
                self.calendar_client.delete_matching_events([target])
                self.calendar_client.create_events([updated])
            return self._format_move_message(target, updated, executed=executed)

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

    def replace_events(self, events: Sequence[CourseEvent]) -> None:
        """Replace the agent's event list and rebuild RAG context."""
        self.events = list(events)
        self._rag_index = CalendarRAGIndex(self.events)

    def append_events(self, events: Sequence[CourseEvent]) -> None:
        """Append new events to the agent's cache."""
        if not events:
            return
        self.events.extend(events)
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
        if self._default_timezone is not None:
            return self._default_timezone
        return timezone.utc

    def _format_add_message(self, event: CourseEvent, *, executed: bool) -> str:
        window = self._format_event_window(event)
        if executed:
            return f"Task added to your calendar: '{event.title}' on {window}. Have fun!"
        return f"[dry-run] Preview — would add '{event.title}' on {window}. Enable calendar writes to apply."

    def _format_delete_message(self, event: CourseEvent, *, executed: bool) -> str:
        window = self._format_event_window(event)
        if executed:
            return f"Removed '{event.title}' scheduled for {window}."
        return f"[dry-run] Preview — would remove '{event.title}' scheduled for {window}."

    def _format_move_message(self, old: CourseEvent, new: CourseEvent, *, executed: bool) -> str:
        old_window = self._format_event_window(old)
        new_window = self._format_event_window(new)
        if executed:
            return f"Moved '{old.title}' from {old_window} to {new_window}. Good luck!"
        return f"[dry-run] Preview — would move '{old.title}' from {old_window} to {new_window}."

    def _format_event_window(self, event: CourseEvent) -> str:
        start = event.start
        end = event.end or event.start
        tz_name = self._tz_display(start.tzinfo)
        fmt = "%b %d @ %I:%M %p"
        start_str = start.strftime(fmt).lstrip("0")
        if end != start:
            end_str = end.strftime("%I:%M %p").lstrip("0")
            return f"{start_str} – {end_str} {tz_name}".strip()
        return f"{start_str} {tz_name}".strip()

    @staticmethod
    def _tz_display(tzinfo: tzinfo | None) -> str:
        if tzinfo is None:
            return ""
        label = getattr(tzinfo, "tzname", None)
        if callable(label):
            label = label(None)
        if not label:
            label = getattr(tzinfo, "zone", "") or getattr(tzinfo, "key", "")
        return label or ""


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
        start = date_parser.isoparse(self.start)
        end = date_parser.isoparse(self.end) if self.end else None
        if end and end <= start:
            end = self._adjust_end_time(start, end)
        return CourseEvent(
            title=self.title,
            category=self.category or EventCategory.OTHER,
            start=start,
            end=end,
            location=self.location,
            description=self.description,
            source_url="agentic_calendar",
        )

    @staticmethod
    def _adjust_end_time(start: datetime, proposed_end: datetime) -> datetime:
        """Heuristic to keep end after start when the LLM omits am/pm."""

        candidate = proposed_end
        # If the user supplied a same-day range like "10-11" without meridiem,
        # the LLM often encodes 11:00 but in the morning. Try bumping by 12h first.
        if candidate.date() == start.date():
            candidate = candidate + timedelta(hours=12)
            if candidate <= start:
                candidate = candidate + timedelta(hours=12)

        if candidate <= start:
            candidate = start + timedelta(hours=1)
        return candidate


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
