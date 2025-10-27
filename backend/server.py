from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone, tzinfo
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Literal

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
from gymscraper import fetch_rsf_occupancy_percent

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from menu_recommender import MenuRecommender


DEFAULT_MENU_SOURCE_URL = "https://dining.berkeley.edu/menus/"
DEFAULT_MENUS_JSON_PATH = "menus.json"
DEFAULT_MENU_RECOMMENDATION_CACHE_MINUTES = 30

try:
    from calhacks import save_to_csv, scrape_course_schedule
except ImportError as exc:  # pragma: no cover - optional scraper dependency
    raise RuntimeError("calhacks.py must be importable to support course scraping") from exc


logger = logging.getLogger("calendar_server")
logging.basicConfig(level=logging.INFO)

COURSE_KEYWORDS = ("course", "syllabus", "schedule", "class")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
TODO_SYSTEM_PROMPT = """
You are a helpful assistant that reads a list of a student's scheduled events for today and suggests
three high-impact focus tasks for them to prioritize. The tasks should be action-oriented and specific,
not generic. If the schedule is light and you cannot find three meaningful tasks, respond with an empty
JSON array instead of inventing filler items.

Return JSON only. Use this schema:
[
  {"text": "<short actionable focus item>"},
  ...
]

The array may contain from zero to three items. Do not include any other fields or commentary.
""".strip()


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
    enable_menu_recommendations: bool = _env_bool("ENABLE_MENU_RECOMMENDATIONS", True)
    menu_source_url: str = os.getenv("MENU_SOURCE_URL", DEFAULT_MENU_SOURCE_URL)
    menus_json: Path = Path(os.getenv("MENUS_JSON", DEFAULT_MENUS_JSON_PATH))
    menu_recommendation_cache_minutes: int = int(
        os.getenv("MENU_RECOMMENDATION_CACHE_MINUTES", str(DEFAULT_MENU_RECOMMENDATION_CACHE_MINUTES))
    )
    menu_recommendation_user_id: str = os.getenv("MENU_RECOMMENDATION_USER_ID", "default-user")


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


class ConfirmationRequest(BaseModel):
    response: str = Field(min_length=1, max_length=200)
    pending_event: dict = Field(description="The pending event data to confirm")


class ChatResponse(BaseModel):
    reply: str
    action: str
    executed: bool
    raw_response: Optional[str] = None
    metadata: dict | None = None
    needs_confirmation: bool = False
    pending_event: Optional[dict] = None


class GmailSyncInfo(BaseModel):
    timestamp: datetime
    event_count: int
    created_count: int
    applied: bool
    errors: List[str] = Field(default_factory=list)


class TodoItem(BaseModel):
    id: str
    text: str


class GymStatus(BaseModel):
    occupancy_percent: Optional[int] = None
    last_updated: Optional[datetime] = None
    error: Optional[str] = None


class MenuRecommendation(BaseModel):
    name: str
    location: Optional[str] = None
    meal: Optional[str] = None
    category: Optional[str] = None
    item_id: Optional[str] = None
    blurb: Optional[str] = None
    hours: List[str] = Field(default_factory=list)
    status: Optional[str] = None
    crowdedness: Optional[dict] = None
    score: Optional[float] = None
    menu_reference: Optional[dict] = None
    nutrition: Optional[dict] = None


class StatusResponse(BaseModel):
    gmail_sync: Optional[GmailSyncInfo] = None
    gym_status: Optional[GymStatus] = None
    menu_recommendation: Optional[MenuRecommendation] = None


class MenuFeedbackRequest(BaseModel):
    item_id: str = Field(min_length=1)
    vote: Literal["upvote", "downvote"]
    user_id: Optional[str] = None


class MenuFeedbackResponse(BaseModel):
    menu_recommendation: Optional[MenuRecommendation] = None


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
        self._lock = asyncio.Lock()
        self._gym_lock = asyncio.Lock()
        self._todo_lock = Lock()
        self._todo_cache: tuple[date, List[TodoItem]] | None = None
        self._todo_cache_expiration: datetime | None = None
        self._todo_cache_ttl = timedelta(minutes=10)
        self._todo_next_refresh: datetime | None = None
        self._csv_source = str(self.config.schedule_csv.resolve())
        self.agent = CalendarAgent(
            events=[],
            calendar_client=self.calendar_client,
            llm_client=self.llm_client,
            default_timezone=self._default_tzinfo,
            on_events_updated=self._handle_events_updated,
        )
        self._load_initial_events()
        self._last_gmail_sync: GmailSyncInfo | None = None
        self._gym_status_cache: GymStatus | None = None
        self._gym_status_expiration: datetime | None = None
        self._gym_cache_ttl = timedelta(minutes=5)
        self._menu_refresh_lock = asyncio.Lock()
        self._menu_lock = asyncio.Lock()
        self._menu_recommender: MenuRecommender | None = None
        self._menu_last_refresh: datetime | None = None
        self._menu_cache: Dict[str, MenuRecommendation] = {}
        self._menu_cache_expiration: Dict[str, datetime] = {}

    async def handle_chat(self, text: str) -> AgentResult:
        async with self._lock:
            return await asyncio.to_thread(self.agent.handle_request, text, dry_run=self.config.dry_run)

    async def handle_confirmation(self, response: str, pending_event_data: dict) -> AgentResult:
        async with self._lock:
            # Convert dict back to CourseEvent
            from agentic_calendar.models import CourseEvent, EventCategory
            from datetime import datetime
            from dateutil import parser as date_parser
            
            pending_event = CourseEvent(
                title=pending_event_data["title"],
                category=EventCategory(pending_event_data["category"]),
                start=date_parser.isoparse(pending_event_data["start"]),
                end=date_parser.isoparse(pending_event_data["end"]) if pending_event_data.get("end") else None,
                location=pending_event_data.get("location"),
                description=pending_event_data.get("description"),
                source_url=pending_event_data.get("source_url"),
            )
            
            return await asyncio.to_thread(
                self.agent.handle_confirmation, 
                response, 
                pending_event, 
                dry_run=self.config.dry_run
            )

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

    async def refresh_menu_data(self, *, force: bool = False) -> None:
        if not self.config.enable_menu_recommendations:
            return
        async with self._menu_refresh_lock:
            first_init = self._menu_last_refresh is None
            now = datetime.now(timezone.utc)
            if (
                not force
                and self._menu_last_refresh
                and now - self._menu_last_refresh < timedelta(minutes=5)
            ):
                return
            if first_init:
                self._reset_menu_memory()
            output_path = self.config.menus_json
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                logger.warning("Could not ensure menu output directory %s", output_path.parent, exc_info=True)
            try:
                from scraper import run_scrape
            except Exception as exc:  # pragma: no cover - dependency missing
                logger.warning("Menu scraper unavailable: %s", exc)
                return
            url = self.config.menu_source_url or DEFAULT_MENU_SOURCE_URL
            try:
                await asyncio.to_thread(run_scrape, url, output_path)
            except Exception:
                logger.exception("Failed to scrape dining menu data from %s", url)
                return
            try:
                from menu_recommender import build_recommender
            except Exception as exc:  # pragma: no cover - dependency missing
                logger.warning("Menu recommender unavailable: %s", exc)
                return
            try:
                recommender = await asyncio.to_thread(build_recommender, output_path)
            except Exception:
                logger.exception("Failed to build menu recommender from %s", output_path)
                return
            async with self._menu_lock:
                self._menu_recommender = recommender
                self._menu_last_refresh = datetime.now(timezone.utc)
                self._menu_cache.clear()
                self._menu_cache_expiration.clear()
            item_count = len(getattr(recommender.kb, "items", []))
            logger.info("Menu recommender initialized with %s items", item_count)

    async def get_menu_recommendation(
        self,
        *,
        force_refresh: bool = False,
        user_id: Optional[str] = None,
    ) -> MenuRecommendation | None:
        if not self.config.enable_menu_recommendations:
            return None
        user_key = user_id or self.config.menu_recommendation_user_id
        now = datetime.now(timezone.utc)
        async with self._menu_lock:
            cached = self._menu_cache.get(user_key)
            expiration = self._menu_cache_expiration.get(user_key)
            if (
                not force_refresh
                and cached
                and expiration
                and now < expiration
            ):
                return cached
            recommender = self._menu_recommender
        if recommender is None:
            await self.refresh_menu_data(force=True)
            async with self._menu_lock:
                recommender = self._menu_recommender
                if recommender is None:
                    return None
        try:
            results = await asyncio.to_thread(
                recommender.recommend,
                "best well-balanced meal today",
                top_k=5,
                user_id=user_key,
            )
        except Exception:
            logger.exception("Failed to generate menu recommendation")
            return None
        if not results:
            return None
        top: Optional[dict] = None
        for candidate in results:
            if candidate:
                top = candidate
                break
        if top is None:
            return None
        rec = MenuRecommendation(
            name=top.get("name") or "Dining hall highlight",
            location=top.get("location"),
            meal=top.get("meal"),
            category=top.get("category"),
            item_id=top.get("item_id"),
            blurb=top.get("blurb"),
            hours=list(top.get("hours") or []),
            status=top.get("status"),
            crowdedness=top.get("crowdedness"),
            score=top.get("score"),
            menu_reference=top.get("menu_reference"),
            nutrition=top.get("nutrition"),
        )
        async with self._menu_lock:
            ttl_minutes = max(1, self.config.menu_recommendation_cache_minutes)
            self._menu_cache[user_key] = rec
            self._menu_cache_expiration[user_key] = now + timedelta(minutes=ttl_minutes)
        return rec

    @staticmethod
    def _reset_menu_memory() -> None:
        storage_root = Path(os.getenv("LETTA_STORAGE_ROOT", ".cache")) / "letta"
        storage_path = storage_root / "feedback.json"
        try:
            storage_path.unlink(missing_ok=True)
        except Exception:
            logger.warning("Failed to reset menu memory at %s", storage_path, exc_info=True)

    async def record_menu_feedback(
        self,
        *,
        item_id: str,
        vote: Literal["upvote", "downvote"],
        user_id: Optional[str] = None,
    ) -> MenuRecommendation | None:
        if not self.config.enable_menu_recommendations:
            raise RuntimeError("Menu recommendations are disabled.")
        user_key = user_id or self.config.menu_recommendation_user_id
        recommender = self._menu_recommender
        if recommender is None:
            raise RuntimeError("Menu recommender is not initialized.")
        vote_value = 1 if vote == "upvote" else -1
        try:
            await asyncio.to_thread(recommender.record_feedback, user_key, item_id, vote_value)
        except KeyError as exc:
            raise KeyError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to record feedback: {exc}") from exc
        async with self._menu_lock:
            self._menu_cache.pop(user_key, None)
            self._menu_cache_expiration.pop(user_key, None)
        return await self.get_menu_recommendation(force_refresh=True, user_id=user_key)

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

    def _handle_events_updated(self, _events: Sequence[CourseEvent]) -> None:
        self._invalidate_todo_cache()

    def _invalidate_todo_cache(self) -> None:
        with self._todo_lock:
            self._todo_cache = None
            self._todo_cache_expiration = None
            self._todo_next_refresh = None

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

    async def get_gym_status(self) -> GymStatus | None:
        now = datetime.now(timezone.utc)
        async with self._gym_lock:
            if (
                self._gym_status_cache
                and self._gym_status_expiration
                and now < self._gym_status_expiration
            ):
                return self._gym_status_cache

            status = await asyncio.to_thread(self._fetch_gym_status)
            self._gym_status_cache = status
            self._gym_status_expiration = now + self._gym_cache_ttl
            return status

    def _fetch_gym_status(self) -> GymStatus:
        timestamp = datetime.now(timezone.utc)
        try:
            percent = fetch_rsf_occupancy_percent()
        except Exception as exc:  # pragma: no cover - scraper best effort
            logger.warning("Gym scraper failed: %s", exc)
            return GymStatus(last_updated=timestamp, error=str(exc))
        return GymStatus(occupancy_percent=percent, last_updated=timestamp)

    async def get_daily_focus_items(self, *, force_refresh: bool = False) -> List[TodoItem]:
        tz_local = self._default_tzinfo or timezone.utc
        now_local = datetime.now(tz_local)
        now_utc = datetime.now(timezone.utc)

        with self._todo_lock:
            cache_entry = self._todo_cache
            cache_expiration = self._todo_cache_expiration
            next_refresh = self._todo_next_refresh
        if (
            not force_refresh
            and cache_entry
            and cache_expiration
            and now_utc < cache_expiration
            and cache_entry[0] == now_local.date()
            and (next_refresh is None or now_utc < next_refresh)
        ):
            return list(cache_entry[1])

        async with self._lock:
            events_snapshot = list(self.agent.events)

        todays_events = self._events_for_date(events_snapshot, now_local.date(), tz_local)
        if not todays_events:
            result = [TodoItem(id="focus-break", text="Done for the day - take a break!")]
        else:
            schedule_summary = self._build_schedule_summary(todays_events, tz_local)
            prompt = (
                f"Today is {now_local.strftime('%A, %B %d, %Y')}.\n"
                f"Timezone: {tz_local}.\n"
                "Here is the user's schedule for today:\n"
                f"{schedule_summary}\n\n"
                "Suggest up to three specific, high-impact action items the user should focus on today."
            )

            try:
                response = await asyncio.to_thread(
                    self.llm_client.chat,
                    [{"role": "user", "content": prompt}],
                    system_prompt=TODO_SYSTEM_PROMPT,
                )
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Failed to generate focus items via LLM: %s", exc)
                result = [TodoItem(id="focus-break", text="Done for the day - take a break!")]
            else:
                focus_items = self._parse_focus_response(response)
                if len(focus_items) >= 3:
                    result = [
                        TodoItem(id=f"focus-{idx}", text=text)
                        for idx, text in enumerate(focus_items[:3], start=1)
                    ]
                else:
                    if focus_items:
                        logger.info("LLM returned fewer than 3 focus items; showing fallback card.")
                    result = [TodoItem(id="focus-break", text="Done for the day - take a break!")]

        with self._todo_lock:
            self._todo_cache = (now_local.date(), result)
            self._todo_cache_expiration = datetime.now(timezone.utc) + self._todo_cache_ttl
            next_midnight_local = datetime.combine(now_local.date() + timedelta(days=1), time.min, tz_local)
            self._todo_next_refresh = next_midnight_local.astimezone(timezone.utc)
        return list(result)

    async def prime_daily_focus_items(self) -> None:
        try:
            await self.get_daily_focus_items(force_refresh=True)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Failed to prime daily focus items")

    def _events_for_date(
        self, events: Sequence[CourseEvent], target_date: date, tz_local: tzinfo
    ) -> List[CourseEvent]:
        start_of_day = datetime.combine(target_date, time.min, tz_local)
        end_of_day = start_of_day + timedelta(days=1)
        todays: List[CourseEvent] = []
        for event in events:
            event_start = self._ensure_timezone(event.start, tz_local)
            if start_of_day <= event_start < end_of_day:
                todays.append(event)
        return todays

    def _build_schedule_summary(self, events: Sequence[CourseEvent], tz_local: tzinfo) -> str:
        parts: List[str] = []
        for event in sorted(events, key=lambda evt: self._ensure_timezone(evt.start, tz_local)):
            start_local = self._ensure_timezone(event.start, tz_local)
            end_local = self._ensure_timezone(event.end, tz_local) if event.end else None
            time_range = start_local.strftime("%I:%M %p").lstrip("0")
            if end_local:
                time_range = f"{time_range} - {end_local.strftime('%I:%M %p').lstrip('0')}"
            location = f" @ {event.location}" if event.location else ""
            parts.append(f"- {time_range}: {event.title}{location}")
        return "\n".join(parts) or "- No scheduled events."

    def _ensure_timezone(self, dt_obj: datetime, tz_local: tzinfo) -> datetime:
        if dt_obj.tzinfo is None:
            return dt_obj.replace(tzinfo=tz_local)
        return dt_obj.astimezone(tz_local)

    def _parse_focus_response(self, response: str) -> List[str]:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("LLM todo response was not valid JSON: %s", response[:200])
            return []

        if not isinstance(data, list):
            logger.warning("LLM todo response was not a list: %s", data)
            return []

        items: List[str] = []
        for entry in data:
            text: Optional[str] = None
            if isinstance(entry, str):
                text = entry
            elif isinstance(entry, dict):
                raw = entry.get("text")
                if raw is not None:
                    text = str(raw)
            if text:
                normalized = text.strip()
                if normalized:
                    items.append(normalized)
        return items


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
    await service.refresh_menu_data(force=True)
    await service.prime_daily_focus_items()
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
        needs_confirmation=result.needs_confirmation,
        pending_event=result.pending_event.model_dump() if result.pending_event else None,
    )


@app.post("/api/confirm", response_model=ChatResponse)
async def confirm(request: ConfirmationRequest) -> ChatResponse:
    try:
        result = await service.handle_confirmation(request.response, request.pending_event)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        reply=result.detail,
        action=result.action.value,
        executed=result.executed,
        raw_response=result.raw_response,
        metadata={"timestamp": datetime.utcnow().isoformat()},
        needs_confirmation=result.needs_confirmation,
        pending_event=result.pending_event.model_dump() if result.pending_event else None,
    )


@app.get("/api/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    gym_status = await service.get_gym_status()
    menu_rec = await service.get_menu_recommendation(user_id=config.menu_recommendation_user_id)
    return StatusResponse(
        gmail_sync=service.get_gmail_sync_info(),
        gym_status=gym_status,
        menu_recommendation=menu_rec,
    )


@app.get("/api/todos", response_model=List[TodoItem])
async def todos() -> List[TodoItem]:
    return await service.get_daily_focus_items()


@app.post("/api/menu/feedback", response_model=MenuFeedbackResponse)
async def menu_feedback(request: MenuFeedbackRequest) -> MenuFeedbackResponse:
    try:
        next_rec = await service.record_menu_feedback(
            item_id=request.item_id,
            vote=request.vote,
            user_id=request.user_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return MenuFeedbackResponse(menu_recommendation=next_rec)
