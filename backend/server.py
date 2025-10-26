from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dateutil import tz

from agentic_calendar import CalendarAgent, GoogleCalendarClient, load_schedule_csv
from agentic_calendar.agent import AgentResult
from agentic_calendar.gmail_client import GmailClient
from agentic_calendar.gmail_ingest import ingest_gmail
from agentic_calendar.llm_client import LavaPaymentsLLMClient
from agentic_calendar.models import CourseEvent

try:
    from calhacks import save_to_csv, scrape_course_schedule
except ImportError as exc:  # pragma: no cover - optional scraper dependency
    raise RuntimeError("calhacks.py must be importable to support course scraping") from exc


logger = logging.getLogger("calendar_server")
logging.basicConfig(level=logging.INFO)

COURSE_KEYWORDS = ("course", "syllabus", "schedule", "class")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)


def _env_bool(key: str, default: bool = False) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_default_start(value: str) -> tuple[int, int]:
    hour_str, minute_str = value.split(":")
    return int(hour_str), int(minute_str)


@dataclass
class ServerConfig:
    schedule_csv: Path = Path(os.getenv("SCHEDULE_CSV", "schedule.csv"))
    calendar_id: Optional[str] = os.getenv("GOOGLE_CALENDAR_ID")
    calendar_credentials: str = os.getenv("CALENDAR_CREDENTIALS", "credentials.json")
    calendar_token: str = os.getenv("CALENDAR_TOKEN", "token.json")
    timezone: str = os.getenv("CALENDAR_TIMEZONE", "America/Los_Angeles")
    default_start: tuple[int, int] = _parse_default_start(os.getenv("DEFAULT_START_TIME", "09:00"))
    default_duration_minutes: int = int(os.getenv("DEFAULT_DURATION_MINUTES", "50"))
    fallback_year: Optional[int] = (
        int(os.getenv("SCHEDULE_FALLBACK_YEAR")) if os.getenv("SCHEDULE_FALLBACK_YEAR") else None
    )
    dry_run: bool = _env_bool("CALENDAR_AGENT_DRY_RUN", True)
    enable_gmail_polling: bool = _env_bool("ENABLE_GMAIL_POLLING", True)
    gmail_poll_interval_seconds: int = int(os.getenv("GMAIL_POLL_INTERVAL_SECONDS", "60"))
    gmail_query: str = os.getenv("GMAIL_QUERY", "label:unread newer_than:1d")
    gmail_credentials: str = os.getenv("GMAIL_CREDENTIALS", "credentials.json")
    gmail_token: str = os.getenv("GMAIL_TOKEN", "gmail_token.json")
    gmail_processed_store: str = os.getenv("GMAIL_PROCESSED_STORE", ".gmail_processed.json")
    llm_provider: str = os.getenv("LAVAPAY_PROVIDER", "openai")
    llm_model: str = os.getenv("LAVAPAY_MODEL", "gpt-4o-mini")
    forward_token: Optional[str] = (
        os.getenv("CALENDAR_AGENT_FORWARD_TOKEN")
        or os.getenv("LAVAPAY_FORWARD_TOKEN")
        or os.getenv("LAVA_FORWARD_TOKEN")
    )
    frontend_origins: Sequence[str] = tuple(
        origin.strip()
        for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    )
    console_oauth: bool = _env_bool("CONSOLE_OAUTH", False)
    course_import_replace: bool = _env_bool("COURSE_IMPORT_REPLACE", False)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    reply: str
    action: str
    executed: bool
    raw_response: Optional[str] = None
    metadata: dict | None = None


class GmailSyncInfo(BaseModel):
    timestamp: datetime
    event_count: int
    created_count: int
    applied: bool
    errors: List[str] = Field(default_factory=list)


class StatusResponse(BaseModel):
    gmail_sync: Optional[GmailSyncInfo] = None


class CourseImportSummary(BaseModel):
    url: str
    events_loaded: int
    warnings: List[str]
    calendar_event_ids: List[str]
    csv_path: str
    executed: bool


class AgentService:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.calendar_client = (
            GoogleCalendarClient(
                calendar_id=config.calendar_id,
                credentials_file=config.calendar_credentials,
                token_file=config.calendar_token,
                dry_run=config.dry_run,
                use_console_oauth=config.console_oauth,
            )
            if config.calendar_id
            else None
        )
        self.llm_client = LavaPaymentsLLMClient(
            forward_token=config.forward_token,
            provider=config.llm_provider,
            model=config.llm_model,
        )
        self._default_tzinfo = tz.gettz(config.timezone) or timezone.utc
        self.agent = CalendarAgent(
            events=[],
            calendar_client=self.calendar_client,
            llm_client=self.llm_client,
            default_timezone=self._default_tzinfo,
        )
        self._lock = asyncio.Lock()
        self._csv_source = str(self.config.schedule_csv.resolve())
        self._load_initial_events()
        self._last_gmail_sync: GmailSyncInfo | None = None

    async def handle_chat(self, text: str) -> AgentResult:
        async with self._lock:
            return await asyncio.to_thread(self.agent.handle_request, text, dry_run=self.config.dry_run)

    def _load_initial_events(self) -> None:
        try:
            report = load_schedule_csv(
                csv_path=self.config.schedule_csv,
                timezone=self.config.timezone,
                default_start_hour=self.config.default_start[0],
                default_start_minute=self.config.default_start[1],
                default_duration_minutes=self.config.default_duration_minutes,
                fallback_year=self.config.fallback_year,
            )
        except FileNotFoundError:
            logger.warning("Schedule CSV %s not found; starting with empty agent state", self.config.schedule_csv)
            return
        self.agent.replace_events(report.events)

    async def reload_events_from_csv(self) -> None:
        report = await asyncio.to_thread(
            load_schedule_csv,
            self.config.schedule_csv,
            self.config.timezone,
            self.config.default_start[0],
            self.config.default_start[1],
            self.config.default_duration_minutes,
            self.config.fallback_year,
        )
        async with self._lock:
            preserved = [event for event in self.agent.events if not self._is_csv_event(event)]
            merged = self._dedupe_events([*report.events, *preserved])
            self.agent.replace_events(merged)

    async def append_events(self, events: Sequence[CourseEvent]) -> int:
        if not events:
            return 0
        async with self._lock:
            existing_keys = {self._event_key(event) for event in self.agent.events}
            new_items = [event for event in events if self._event_key(event) not in existing_keys]
            if not new_items:
                return 0
            self.agent.append_events(new_items)
            return len(new_items)

    async def import_course(self, url: str) -> CourseImportSummary:
        events = await asyncio.to_thread(scrape_course_schedule, url)
        await asyncio.to_thread(save_to_csv, events, str(self.config.schedule_csv))
        report = await asyncio.to_thread(
            load_schedule_csv,
            self.config.schedule_csv,
            self.config.timezone,
            self.config.default_start[0],
            self.config.default_start[1],
            self.config.default_duration_minutes,
            self.config.fallback_year,
        )

        async with self._lock:
            preserved = [event for event in self.agent.events if not self._is_csv_event(event)]
            merged = self._dedupe_events([*report.events, *preserved])
            self.agent.replace_events(merged)

        created_ids: List[str] = []
        executed = False
        if self.calendar_client and not self.config.dry_run:
            if self.config.course_import_replace:
                await asyncio.to_thread(self.calendar_client.delete_matching_events, report.events)
            created_ids = await asyncio.to_thread(self.calendar_client.create_events, report.events)
            executed = bool(created_ids)

        return CourseImportSummary(
            url=url,
            events_loaded=len(report.events),
            warnings=report.warnings,
            calendar_event_ids=created_ids,
            csv_path=str(self.config.schedule_csv),
            executed=executed,
        )

    async def run_gmail_ingest(self) -> None:
        if not self.config.enable_gmail_polling or not self.calendar_client:
            return
        gmail_client = GmailClient(
            credentials_file=self.config.gmail_credentials,
            token_file=self.config.gmail_token,
            use_console_oauth=self.config.console_oauth,
        )
        result = await asyncio.to_thread(
            ingest_gmail,
            gmail_client=gmail_client,
            llm_client=self.llm_client,
            calendar_client=self.calendar_client if not self.config.dry_run else None,
            query=self.config.gmail_query,
            timezone=self.config.timezone,
            processed_store=self.config.gmail_processed_store,
            apply=not self.config.dry_run,
        )
        self._record_gmail_sync(result, applied=not self.config.dry_run)
        if result.events:
            created = await self.append_events(result.events)
            logger.info("Gmail ingest added %s new events; created_ids=%s", created, len(result.created_event_ids))

    @staticmethod
    def _event_key(event: CourseEvent) -> tuple[str, str]:
        return (event.title.lower(), event.start.isoformat())

    @staticmethod
    def _dedupe_events(events: Sequence[CourseEvent]) -> List[CourseEvent]:
        seen = {}
        for event in events:
            seen[(event.title.lower(), event.start.isoformat())] = event
        return list(seen.values())

    def _is_csv_event(self, event: CourseEvent) -> bool:
        if not event.source_url:
            return False
        try:
            return str(Path(event.source_url).resolve()) == self._csv_source
        except Exception:
            return False

    def _record_gmail_sync(self, result, *, applied: bool) -> None:
        self._last_gmail_sync = GmailSyncInfo(
            timestamp=datetime.now(timezone.utc),
            event_count=len(result.events),
            created_count=len(result.created_event_ids),
            applied=applied and bool(result.created_event_ids),
            errors=result.errors,
        )

    def get_gmail_sync_info(self) -> GmailSyncInfo | None:
        return self._last_gmail_sync


config = ServerConfig()
service = AgentService(config)
app = FastAPI(title="Calendar Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.frontend_origins),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


def _extract_course_link(message: str) -> Optional[str]:
    lowered = message.lower()
    if not any(keyword in lowered for keyword in COURSE_KEYWORDS):
        return None
    match = URL_RE.search(message)
    if not match:
        return None
    url = match.group(0)
    return url.rstrip(").,")


@app.on_event("startup")
async def _startup() -> None:
    if not config.enable_gmail_polling:
        logger.info("Gmail polling disabled (set ENABLE_GMAIL_POLLING=true to enable).")
        return
    if config.dry_run:
        logger.info("Gmail polling skipped because CALENDAR_AGENT_DRY_RUN=true.")
        return
    if service.calendar_client is None:
        logger.warning("Gmail polling enabled but Google Calendar is not configured; skipping.")
        return

    async def _poll_loop() -> None:
        logger.info("Starting Gmail polling every %s seconds", config.gmail_poll_interval_seconds)
        while True:
            try:
                await service.run_gmail_ingest()
            except Exception:  # pragma: no cover - defensive logging
                logger.exception("Gmail ingest run failed")
            await asyncio.sleep(config.gmail_poll_interval_seconds)

    asyncio.create_task(_poll_loop())


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message is required.")

    course_link = _extract_course_link(message)
    if course_link:
        try:
            summary = await service.import_course(course_link)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Failed to import course: {exc}") from exc
        reply = (
            f"Imported {summary.events_loaded} events from {course_link}. "
            f"{'Synced to Google Calendar.' if summary.executed else 'Dry-run mode; no Calendar changes.'}"
        )
        if summary.warnings:
            reply += f" Warnings: {'; '.join(summary.warnings)}"
        return ChatResponse(
            reply=reply,
            action="import_course",
            executed=summary.executed,
            metadata=summary.model_dump(),
        )

    try:
        result = await service.handle_chat(message)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        reply=result.detail,
        action=result.action.value,
        executed=result.executed,
        raw_response=result.raw_response,
        metadata={"timestamp": datetime.utcnow().isoformat()},
    )


@app.get("/api/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    return StatusResponse(gmail_sync=service.get_gmail_sync_info())
