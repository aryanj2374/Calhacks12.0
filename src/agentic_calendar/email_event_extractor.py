from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import List

from dateutil import parser as date_parser

from .gmail_client import GmailMessage
from .llm_client import LavaPaymentsLLMClient
from .models import CourseEvent, EventCategory


class EmailEventExtractor:
    """
    Uses an LLM to convert raw email bodies into structured `CourseEvent` objects.

    This version mirrors the earlier logic that successfully captured HW4 deadlines and the
    CalHacks opening ceremony before the stricter promotional filters were added.
    """

    SYSTEM_PROMPT = """
You analyze email bodies and extract every discrete event or deadline that the recipient
should add to their calendar. Always respond with valid JSON only:
{
  "events": [
    {
      "title": "string",
      "category": "lecture|discussion|lab|midterm|final|review|homework|project|quiz|other",
      "start": "ISO 8601 datetime",
      "end": "ISO 8601 datetime or null",
      "location": "string or null",
      "description": "string or null"
    }
  ]
}
Guidelines:
- Only include entries when the email clearly states a date (absolute or relative) and a time.
- Resolve relative phrases like "today" or "tomorrow" using the provided `current_datetime`
  and `email_received_at` values.
- Prefer concise titles; use the email subject if no better title exists.
- When the email lists multiple sessions (e.g., a hackathon schedule or newsletter with deadlines),
  output one event per line item.
- Ignore obvious marketing blasts (coupon codes, generic sales) unless the email explicitly says the
  recipient is registered/committed.
"""

    PROMO_KEYWORDS = [
        "promotion",
        "coupon",
        "discount",
        "sale",
        "tickets on sale",
        "buy tickets",
        "pre-sale",
        "early bird",
        "register now",
        "sign up",
        "limited time",
        "apply now",
    ]

    COMMITMENT_KEYWORDS = [
        "registration confirmed",
        "you're registered",
        "your ticket",
        "order confirmation",
        "receipt",
        "due",
        "deadline",
        "assignment",
        "homework",
        "meeting",
        "opening ceremony",
        "check-in",
        "see you",
        "starts at",
        "team formation",
        "workshop",
        "hackathon",
        "ceremony",
    ]

    def __init__(
        self,
        llm_client: LavaPaymentsLLMClient,
        *,
        timezone_name: str = "America/Los_Angeles",
    ):
        self.llm_client = llm_client
        self.timezone_name = timezone_name
        self.debug = bool(os.environ.get("EMAIL_DEBUG"))

    def extract_events(self, message: GmailMessage) -> List[CourseEvent]:
        combined = self._combine_text(message)
        promo = self._looks_promotional(combined)
        commitment = self._has_commitment_signal(combined)

        if self.debug:
            print(
                "[email-debug]",
                {
                    "subject": message.subject,
                    "promotional": promo,
                    "commitment": commitment,
                },
            )

        # Skip obvious marketing blasts unless they clearly mention a commitment.
        if promo and not commitment:
            return []

        user_prompt = self._build_prompt(message)
        response = self.llm_client.chat(
            [{"role": "user", "content": user_prompt}],
            system_prompt=self.SYSTEM_PROMPT.strip(),
        )

        payload = _parse_json(response)
        events_data = payload.get("events", [])
        results: List[CourseEvent] = []
        for item in events_data:
            try:
                results.append(self._to_course_event(item, message))
            except ValueError:
                continue
        return results

    def _combine_text(self, message: GmailMessage) -> str:
        return "\n".join(filter(None, [message.subject, message.snippet, message.body_text]))

    def _build_prompt(self, message: GmailMessage) -> str:
        now = datetime.now(timezone.utc).isoformat()
        return (
            f"current_datetime: {now}\n"
            f"email_received_at: {message.received_at.isoformat()}\n"
            f"timezone: {self.timezone_name}\n"
            f"email_subject: {message.subject}\n"
            f"email_sender: {message.sender}\n"
            f"email_snippet: {message.snippet}\n"
            f"email_body:\n{message.body_text}\n"
        )

    def _to_course_event(self, data: dict, message: GmailMessage) -> CourseEvent:
        title = data.get("title") or message.subject or "Email Event"
        category = _coerce_category(data.get("category"))
        start_text = data.get("start")
        if not start_text:
            raise ValueError("missing start")
        start = date_parser.isoparse(start_text)
        end_text = data.get("end")
        end = date_parser.isoparse(end_text) if end_text else None
        location = data.get("location") or None
        desc_parts = [data.get("description") or "", f"Source email: {message.subject}"]
        description = "\n".join(part for part in desc_parts if part).strip() or None
        return CourseEvent(
            title=title,
            category=category,
            start=start,
            end=end,
            location=location,
            description=description,
            source_url=f"gmail:{message.id}",
        )

    def _looks_promotional(self, text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in self.PROMO_KEYWORDS)

    def _has_commitment_signal(self, text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in self.COMMITMENT_KEYWORDS)


def _coerce_category(value) -> EventCategory:
    if isinstance(value, EventCategory):
        return value
    if isinstance(value, str):
        try:
            return EventCategory(value.lower())
        except ValueError:
            return EventCategory.OTHER
    return EventCategory.OTHER


def _parse_json(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
