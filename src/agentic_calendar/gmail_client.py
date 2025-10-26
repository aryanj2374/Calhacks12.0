from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import List, Sequence

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


@dataclass
class GmailMessage:
    """Simplified representation of a Gmail message used for event extraction."""

    id: str
    thread_id: str
    subject: str
    sender: str
    snippet: str
    received_at: datetime
    body_text: str


class GmailClient:
    """Thin Gmail wrapper focused on fetching recent messages via OAuth."""

    def __init__(
        self,
        *,
        credentials_file: str = "credentials.json",
        token_file: str = "gmail_token.json",
        user_id: str = "me",
        use_console_oauth: bool = False,
    ):
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.user_id = user_id
        self.use_console_oauth = use_console_oauth
        self._service = None

    def ensure_authenticated(self) -> None:
        if self._service is None:
            creds = self._load_credentials()
            self._service = build("gmail", "v1", credentials=creds)

    def fetch_messages(self, *, query: str, max_results: int | None = None) -> List[GmailMessage]:
        """Return Gmail messages (decoded text bodies) that match the query."""

        if max_results is not None and max_results <= 0:
            return []

        self.ensure_authenticated()
        assert self._service is not None

        results: List[GmailMessage] = []
        next_page_token: str | None = None
        batch_size = 100

        while True:
            effective_limit = batch_size
            if max_results is not None:
                remaining = max_results - len(results)
                if remaining <= 0:
                    break
                effective_limit = min(batch_size, remaining)

            try:
                response = (
                    self._service.users()
                    .messages()
                    .list(
                        userId=self.user_id,
                        maxResults=effective_limit,
                        q=query,
                        pageToken=next_page_token,
                    )
                    .execute()
                )
            except HttpError as exc:  # pragma: no cover - network error path
                raise RuntimeError(f"Failed to list Gmail messages: {exc}") from exc

            message_refs = response.get("messages", []) or []
            for ref in message_refs:
                msg = self._fetch_single(ref["id"])
                if msg:
                    results.append(msg)
                if max_results is not None and len(results) >= max_results:
                    break

            next_page_token = response.get("nextPageToken")
            if not next_page_token or (max_results is not None and len(results) >= max_results):
                break

        return results

    def _fetch_single(self, message_id: str) -> GmailMessage | None:
        assert self._service is not None
        try:
            response = (
                self._service.users()
                .messages()
                .get(userId=self.user_id, id=message_id, format="full")
                .execute()
            )
        except HttpError as exc:  # pragma: no cover - network error path
            raise RuntimeError(f"Failed to fetch Gmail message '{message_id}': {exc}") from exc

        payload = response.get("payload", {})
        headers = {item["name"].lower(): item["value"] for item in payload.get("headers", [])}
        subject = headers.get("subject", "(no subject)")
        sender = headers.get("from", "(unknown sender)")

        received = _parse_received_datetime(headers) or _parse_internal_date(response.get("internalDate"))
        snippet = response.get("snippet", "")
        body_text = _extract_body_text(payload)

        return GmailMessage(
            id=response.get("id", message_id),
            thread_id=response.get("threadId", ""),
            subject=subject,
            sender=sender,
            snippet=snippet,
            received_at=received or datetime.now(timezone.utc),
            body_text=body_text,
        )

    def _load_credentials(self) -> Credentials:
        creds: Credentials | None = None
        try:
            creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
        except FileNotFoundError:
            creds = None
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(self.credentials_file, SCOPES)
                creds = flow.run_local_server(port=0, open_browser=not self.use_console_oauth)
            with open(self.token_file, "w", encoding="utf-8") as handle:
                handle.write(creds.to_json())
        return creds


def _parse_received_datetime(headers: dict[str, str]) -> datetime | None:
    date_header = headers.get("date")
    if not date_header:
        return None
    try:
        parsed = parsedate_to_datetime(date_header)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (ValueError, TypeError):
        return None


def _parse_internal_date(internal: str | None) -> datetime | None:
    if not internal:
        return None
    try:
        millis = int(internal)
    except ValueError:
        return None
    return datetime.fromtimestamp(millis / 1000, tz=timezone.utc)


def _extract_body_text(payload: dict) -> str:
    """Extract the first text/plain or text/html part from the Gmail payload."""

    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {})
    data = body.get("data")

    if mime_type.startswith("multipart/"):
        parts: Sequence[dict] = payload.get("parts", []) or []
        for part in parts:
            text = _extract_body_text(part)
            if text:
                return text
        return ""

    if mime_type in {"text/plain", "text/html"} and data:
        decoded = base64.urlsafe_b64decode(data.encode("utf-8"))
        text = decoded.decode("utf-8", errors="ignore")
        if mime_type == "text/html":
            return _strip_html(text)
        return text

    return ""


def _strip_html(html_text: str) -> str:
    """Crude HTML tag stripper to avoid bringing in heavy dependencies."""

    inside = False
    output_chars: list[str] = []
    for ch in html_text:
        if ch == "<":
            inside = True
            continue
        if ch == ">":
            inside = False
            output_chars.append(" ")
            continue
        if not inside:
            output_chars.append(ch)
    return " ".join("".join(output_chars).split())
