from __future__ import annotations

import csv
from datetime import datetime, timedelta
import re
from pathlib import Path
from typing import Dict, Iterable

from dateutil import parser as date_parser
from dateutil import tz

from .models import CourseEvent, EventCategory, ExtractionReport

CATEGORY_HINTS: list[tuple[EventCategory, set[str]]] = [
    (EventCategory.FINAL, {"final"}),
    (EventCategory.MIDTERM, {"midterm", "mid-term", "exam", "mt"}),
    (EventCategory.QUIZ, {"quiz"}),
    (EventCategory.PROJECT, {"project", "proj"}),
    (EventCategory.HOMEWORK, {"homework", "hw", "assignment", "problem set", "pset", "due"}),
    (EventCategory.REVIEW, {"review", "practice"}),
    (EventCategory.LAB, {"lab"}),
    (EventCategory.DISCUSSION, {"discussion", "section", "disc"}),
    (EventCategory.LECTURE, {"lecture", "class", "topic", "lec"}),
]

CATEGORY_FALLBACKS: dict[EventCategory, tuple[int, int, int]] = {
    EventCategory.LECTURE: (13, 0, 80),
    EventCategory.DISCUSSION: (13, 0, 50),
    EventCategory.LAB: (14, 0, 120),
    EventCategory.HOMEWORK: (23, 59, 15),
    EventCategory.PROJECT: (23, 59, 15),
    EventCategory.QUIZ: (17, 0, 60),
    EventCategory.MIDTERM: (19, 0, 120),
    EventCategory.FINAL: (8, 0, 180),
    EventCategory.REVIEW: (15, 0, 90),
}

TIME_RANGE_RE = re.compile(
    r"(?P<start>\d{1,2}(?::\d{2})?\s*(?:am|pm))\s*(?:[-–]|to)\s*(?P<end>\d{1,2}(?::\d{2})?\s*(?:am|pm))",
    re.IGNORECASE,
)
TIME_SINGLE_RE = re.compile(r"(?P<single>\d{1,2}(?::\d{2})?\s*(?:am|pm))", re.IGNORECASE)
SIMPLE_RANGE_RE = re.compile(
    r"^\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:[-–]|to)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$",
    re.IGNORECASE,
)
SIMPLE_SINGLE_PLAIN_RE = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?\s*$")
YEAR_IN_TEXT_RE = re.compile(r"\d{4}")
TITLE_MAX_LENGTH = 80


def load_schedule_csv(
    csv_path: str | Path,
    timezone: str = "America/Los_Angeles",
    default_start_hour: int = 9,
    default_start_minute: int = 0,
    default_duration_minutes: int = 50,
    fallback_year: int | None = None,
) -> ExtractionReport:
    """Load course events from a CSV file and return an ExtractionReport."""

    csv_path = Path(csv_path)
    tzinfo = tz.gettz(timezone)
    if tzinfo is None:
        raise ValueError(f"Unknown timezone '{timezone}'.")
    events: list[CourseEvent] = []
    warnings: list[str] = []
    raw_rows: list[str] = []

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return ExtractionReport(
                source_url=str(csv_path),
                events=[],
                warnings=["CSV file is missing a header row."],
                raw_blocks=[],
            )

        for idx, row in enumerate(reader, start=2):
            raw_rows.append(str(row))
            try:
                event = _row_to_event(
                    row=row,
                    tzinfo=tzinfo,
                    default_start_hour=default_start_hour,
                    default_start_minute=default_start_minute,
                    default_duration_minutes=default_duration_minutes,
                    source=str(csv_path),
                    fallback_year=fallback_year,
                )
            except ValueError as exc:
                warnings.append(f"Row {idx}: {exc}")
                continue
            events.append(event)

    if not events:
        warnings.append("CSV file did not yield any events. Check column names and formats.")

    return ExtractionReport(source_url=str(csv_path), events=events, warnings=warnings, raw_blocks=raw_rows)


def _row_to_event(
    row: Dict[str, str],
    tzinfo,
    default_start_hour: int,
    default_start_minute: int,
    default_duration_minutes: int,
    source: str,
    fallback_year: int | None,
) -> CourseEvent:
    normalized = {(_clean_key(key)): (value or "").strip() for key, value in row.items()}

    date_text = _pop_first(normalized, ["date", "day"])
    if not date_text:
        raise ValueError("Missing required 'date' column.")
    if not YEAR_IN_TEXT_RE.search(date_text):
        if fallback_year is None:
            raise ValueError(f"Invalid date '{date_text}': expected a year (YYYY).")
        date_text = f"{date_text} {fallback_year}"

    title = _pop_first(normalized, ["title", "event", "name", "topic"])
    description = _pop_first(normalized, ["description", "notes", "summary"])
    type_hint = _pop_first(normalized, ["type", "category"])
    location = _pop_first(normalized, ["location", "room"]) or None
    duration_override = _pop_first(normalized, ["duration", "duration_minutes"])
    start_time_text = _pop_first(normalized, ["start_time", "start", "time"])
    end_time_text = _pop_first(normalized, ["end_time", "end"])

    if not title:
        title = _derive_title(type_hint, description)

    inferred_start, inferred_end = _infer_times(description)
    if not start_time_text and inferred_start:
        start_time_text = inferred_start
    if not end_time_text and inferred_end:
        end_time_text = inferred_end

    category = _resolve_category(type_hint, title, description)
    fallback_start_hour, fallback_start_minute, fallback_duration_minutes = _resolve_defaults(
        category,
        default_start_hour,
        default_start_minute,
        default_duration_minutes,
    )

    start_time_text, end_time_text = _normalize_time_inputs(start_time_text, end_time_text, fallback_start_hour)

    start = _compose_datetime(date_text, start_time_text, tzinfo, fallback_start_hour, fallback_start_minute)
    end = _compose_end_datetime(
        start,
        end_time_text=end_time_text,
        duration_override=duration_override,
        tzinfo=tzinfo,
        default_duration_minutes=fallback_duration_minutes,
    )

    return CourseEvent(
        title=title,
        category=category,
        start=start,
        end=end,
        location=location,
        description=description or None,
        source_url=source,
    )

def _resolve_defaults(
    category: EventCategory,
    fallback_hour: int,
    fallback_minute: int,
    fallback_duration: int,
) -> tuple[int, int, int]:
    override = CATEGORY_FALLBACKS.get(category)
    if override:
        return override
    return fallback_hour, fallback_minute, fallback_duration


def _compose_datetime(
    date_text: str,
    time_text: str | None,
    tzinfo,
    default_start_hour: int,
    default_start_minute: int,
) -> datetime:
    try:
        date_part = date_parser.parse(date_text, fuzzy=True)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid date '{date_text}': {exc}") from exc

    if time_text:
        try:
            dt = date_parser.parse(
                f"{date_part.strftime('%Y-%m-%d')} {time_text}",
                fuzzy=True,
                default=date_part,
            )
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid time '{time_text}': {exc}") from exc
    else:
        dt = date_part.replace(hour=default_start_hour, minute=default_start_minute, second=0, microsecond=0)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tzinfo)
    return dt


def _compose_end_datetime(
    start: datetime,
    end_time_text: str | None,
    duration_override: str | None,
    tzinfo,
    default_duration_minutes: int,
) -> datetime:
    if end_time_text:
        try:
            end = date_parser.parse(
                f"{start.strftime('%Y-%m-%d')} {end_time_text}",
                fuzzy=True,
                default=start,
            )
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"Invalid end time '{end_time_text}': {exc}") from exc
        if end.tzinfo is None:
            end = end.replace(tzinfo=tzinfo)
        return end

    minutes = default_duration_minutes
    if duration_override:
        try:
            minutes = int(float(duration_override))
        except ValueError:
            raise ValueError(f"Duration must be numeric; got '{duration_override}'.")
    return start + timedelta(minutes=minutes)


def _normalize_time_inputs(
    start_text: str | None,
    end_text: str | None,
    fallback_hour: int,
) -> tuple[str | None, str | None]:
    if not start_text:
        return start_text, end_text

    trimmed = start_text.strip()
    range_match = SIMPLE_RANGE_RE.match(trimmed)
    if range_match and not end_text:
        start_hour = int(range_match.group(1))
        start_minute = int(range_match.group(2) or 0)
        start_meridiem = range_match.group(3)
        end_hour = int(range_match.group(4))
        end_minute = int(range_match.group(5) or 0)
        end_meridiem = range_match.group(6) or start_meridiem

        start_norm = _format_hour(start_hour, start_minute, start_meridiem, fallback_hour)
        end_norm = _format_hour(end_hour, end_minute, end_meridiem, fallback_hour)
        return start_norm, end_norm

    single_plain = SIMPLE_SINGLE_PLAIN_RE.match(trimmed)
    if single_plain:
        hour = int(single_plain.group(1))
        minute = int(single_plain.group(2) or 0)
        return _format_hour(hour, minute, None, fallback_hour), end_text

    return start_text, end_text


def _format_hour(hour: int, minute: int, meridiem: str | None, fallback_hour: int) -> str:
    original_hour = hour
    if meridiem:
        meridiem = meridiem.lower()
        if meridiem == "pm" and hour < 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
    else:
        if fallback_hour >= 12 and hour < 12:
            hour += 12
        if fallback_hour < 12 and hour == 12:
            hour = 0
        # If fallback is morning but the derived hour ends up before fallback, assume pm
        if fallback_hour >= 12 and hour < fallback_hour - 6:
            hour += 12
    # Clamp to 0-23 window
    hour %= 24
    return f"{hour:02d}:{minute:02d}"


def _resolve_category(*texts: str | None) -> EventCategory:
    for text in texts:
        if not text:
            continue
        normalized = text.lower()
        for category in EventCategory:
            if normalized == category.value or normalized == category.name.lower():
                return category
    combined = " ".join(filter(None, texts)).lower()
    for category, hints in CATEGORY_HINTS:
        if any(keyword in combined for keyword in hints):
            return category
    return EventCategory.OTHER


def _derive_title(type_hint: str | None, description: str | None) -> str:
    snippet = ""
    if description:
        snippet = description.strip().splitlines()[0].strip(" \"'")
    if len(snippet) > TITLE_MAX_LENGTH:
        snippet = f"{snippet[: TITLE_MAX_LENGTH - 3]}..."
    if type_hint:
        prefix = type_hint.strip().title()
    else:
        prefix = ""
    if snippet and prefix:
        return f"{prefix}: {snippet}"
    if snippet:
        return snippet
    if prefix:
        return prefix
    return "Course Event"


def _infer_times(description: str | None) -> tuple[str | None, str | None]:
    if not description:
        return None, None
    range_match = TIME_RANGE_RE.search(description)
    if range_match:
        return range_match.group("start"), range_match.group("end")
    single_match = TIME_SINGLE_RE.search(description)
    if single_match:
        return single_match.group("single"), None
    return None, None


def _pop_first(store: Dict[str, str], keys: Iterable[str]) -> str:
    for key in keys:
        if key in store and store[key]:
            return store[key]
    return ""


def _clean_key(key: str | None) -> str:
    return (key or "").strip().lower()
