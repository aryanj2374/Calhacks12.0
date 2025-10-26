from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set

from .calendar_client import GoogleCalendarClient
from .email_event_extractor import EmailEventExtractor
from .gmail_client import GmailClient, GmailMessage
from .llm_client import LavaPaymentsLLMClient
from .models import CourseEvent


@dataclass
class GmailIngestResult:
    """Summary of a Gmail ingest run."""

    events: List[CourseEvent]
    created_event_ids: List[str]
    processed_message_ids: List[str]
    errors: List[str]

    @property
    def new_event_count(self) -> int:
        return len(self.events)


def ingest_gmail(
    *,
    gmail_client: GmailClient,
    llm_client: LavaPaymentsLLMClient,
    calendar_client: GoogleCalendarClient | None = None,
    query: str = "label:unread newer_than:1d",
    timezone: str = "America/Los_Angeles",
    processed_store: str | Path = ".gmail_processed.json",
    apply: bool = False,
) -> GmailIngestResult:
    """
    Fetch Gmail messages, extract events with the LLM, and optionally create them on Calendar.
    """

    store = ProcessedMessageStore(Path(processed_store))

    raw_messages = gmail_client.fetch_messages(query=query)
    new_messages = store.filter_new(raw_messages)
    if not new_messages:
        return GmailIngestResult(events=[], created_event_ids=[], processed_message_ids=[], errors=[])

    extractor = EmailEventExtractor(
        llm_client,
        timezone_name=timezone,
    )

    events: List[CourseEvent] = []
    errors: List[str] = []
    for message in new_messages:
        try:
            events.extend(extractor.extract_events(message))
        except Exception as exc:  # pragma: no cover - defensive logging
            errors.append(f"{message.subject}: {exc}")

    created: List[str] = []
    if events and apply:
        if calendar_client is None:
            errors.append("Calendar client is not configured; skipping Calendar writes.")
        else:
            created = calendar_client.create_events(events)
            store.mark_processed(new_messages)


    return GmailIngestResult(
        events=events,
        created_event_ids=created,
        processed_message_ids=[msg.id for msg in new_messages],
        errors=errors,
    )


class ProcessedMessageStore:
    """Tracks Gmail message IDs that have already been ingested."""

    def __init__(self, path: Path):
        self.path = path
        self.ids: Set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = self.path.read_text(encoding="utf-8")
            if data:
                items = json.loads(data)
            else:
                items = []
            if isinstance(items, list):
                self.ids = set(str(item) for item in items)
        except Exception:  # pragma: no cover - ignore corrupted store
            self.ids = set()

    def filter_new(self, messages: Iterable[GmailMessage]) -> List[GmailMessage]:
        return [message for message in messages if message.id not in self.ids]

    def mark_processed(self, messages: Iterable[GmailMessage]) -> None:
        updated = False
        for message in messages:
            if message.id not in self.ids:
                self.ids.add(message.id)
                updated = True
        if updated:
            self.path.write_text(json.dumps(sorted(self.ids)), encoding="utf-8")
