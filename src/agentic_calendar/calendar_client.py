from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

from datetime import timedelta
from urllib.parse import urlparse

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow

from .models import CourseEvent

SCOPES = ["https://www.googleapis.com/auth/calendar"]


class GoogleCalendarClient:
    """Thin wrapper around the Google Calendar API with a dry-run option."""

    def __init__(
        self,
        calendar_id: str,
        credentials_file: str = "credentials.json",
        token_file: str = "token.json",
        dry_run: bool = False,
        use_console_oauth: bool = False,
    ):
        self.calendar_id = calendar_id
        self.credentials_file = Path(credentials_file)
        self.token_file = Path(token_file)
        self.dry_run = dry_run
        self.use_console_oauth = use_console_oauth
        self._service = None

    def ensure_authenticated(self) -> None:
        if self._service is None:
            creds = self._load_credentials()
            self._service = build("calendar", "v3", credentials=creds)

    def create_events(self, events: Sequence[CourseEvent]) -> List[str]:
        if not events:
            return []
        if self.dry_run:
            return [self._format_preview(event) for event in events]

        self.ensure_authenticated()
        assert self._service is not None

        created_ids: List[str] = []
        for event in events:
            body = self._to_calendar_body(event)
            try:
                response = (
                    self._service.events()
                    .insert(calendarId=self.calendar_id, body=body, sendUpdates="all")
                    .execute()
                )
                created_ids.append(response.get("id", ""))
            except HttpError as exc:  # pragma: no cover - network failure path
                raise RuntimeError(f"Failed to create event '{event.title}': {exc}") from exc
        return created_ids

    def delete_matching_events(self, events: Sequence[CourseEvent]) -> int:
        if not events:
            return 0
        self.ensure_authenticated()
        assert self._service is not None

        deleted = 0
        for event in events:
            window_start = (event.start - timedelta(minutes=1)).isoformat()
            window_end = ((event.end or event.start) + timedelta(minutes=1)).isoformat()
            results = (
                self._service.events()
                .list(
                    calendarId=self.calendar_id,
                    q=event.title,
                    timeMin=window_start,
                    timeMax=window_end,
                    singleEvents=True,
                    orderBy="startTime",
                )
                .execute()
            )
            for item in results.get("items", []):
                if item.get("summary") == event.title:
                    self._service.events().delete(calendarId=self.calendar_id, eventId=item["id"]).execute()
                    deleted += 1
        return deleted

    def _load_credentials(self) -> Credentials:
        creds: Credentials | None = None
        if self.token_file.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_file), SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_file), SCOPES)
                creds = flow.run_local_server(port=0, open_browser=not self.use_console_oauth)
            self.token_file.write_text(creds.to_json())
        return creds

    def _to_calendar_body(self, event: CourseEvent) -> dict:
        end = event.end or event.start
        start_payload = {
            "summary": event.title,
            "description": event.description or "",
            "location": event.location or "",
            "start": {
                "dateTime": event.start.isoformat(),
            },
            "end": {
                "dateTime": end.isoformat(),
            },
        }
        start_tz = self._extract_timezone_name(event.start.tzinfo)
        if start_tz:
            start_payload["start"]["timeZone"] = start_tz
        end_tz = self._extract_timezone_name(end.tzinfo)
        if end_tz:
            start_payload["end"]["timeZone"] = end_tz
        if event.source_url and self._is_http_url(event.source_url):
            start_payload["source"] = {"title": "Course Schedule", "url": event.source_url}
        return start_payload

    @staticmethod
    def _format_preview(event: CourseEvent) -> str:
        payload = {
            "title": event.title,
            "category": event.category.value,
            "start": event.start.isoformat(),
            "end": (event.end or event.start).isoformat(),
            "source": event.source_url,
        }
        return json.dumps(payload, indent=2)

    @staticmethod
    def _is_http_url(value: str) -> bool:
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _extract_timezone_name(tzinfo) -> str | None:
        if tzinfo is None:
            return None
        key = getattr(tzinfo, "key", None)
        if key:
            return key
        zone = getattr(tzinfo, "zone", None)
        if zone:
            return zone
        return None
