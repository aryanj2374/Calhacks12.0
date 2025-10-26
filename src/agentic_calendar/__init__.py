"""Agentic course calendar package."""

from .agent import CalendarAction, CalendarAgent
from .calendar_client import GoogleCalendarClient
from .gmail_client import GmailClient, GmailMessage
from .email_event_extractor import EmailEventExtractor
from .csv_loader import load_schedule_csv
from .models import CourseEvent, EventCategory, ExtractionReport

__all__ = [
    "CalendarAction",
    "CalendarAgent",
    "EmailEventExtractor",
    "GmailClient",
    "GmailMessage",
    "CourseEvent",
    "EventCategory",
    "ExtractionReport",
    "GoogleCalendarClient",
    "load_schedule_csv",
]
