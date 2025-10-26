from __future__ import annotations

import argparse
from typing import List

from rich.console import Console
from rich.table import Table

from .calendar_client import GoogleCalendarClient
from .csv_loader import load_schedule_csv
from .models import CourseEvent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load course events from CSV and push them to Google Calendar")
    parser.add_argument(
        "--csv",
        default="schedule.csv",
        help="Path to the CSV file containing course events (default: schedule.csv)",
    )
    parser.add_argument("--calendar", required=True, help="Google Calendar ID (or primary)")
    parser.add_argument("--credentials", default="credentials.json", help="Path to OAuth client credentials")
    parser.add_argument("--token", default="token.json", help="Path to cached OAuth token")
    parser.add_argument("--timezone", default="America/Los_Angeles", help="IANA timezone for parsed dates")
    parser.add_argument("--duration", type=int, default=50, help="Default event length in minutes when no end time is provided")
    parser.add_argument(
        "--default-start",
        default="09:00",
        help="Fallback start time (HH:MM, 24h) when the schedule omits explicit times",
    )
    parser.add_argument("--max-events", type=int, default=100, help="Limit events pushed to Calendar")
    parser.add_argument(
        "--csv-year",
        type=int,
        default=None,
        help="Default year to apply when the CSV dates omit the year (e.g., 2025)",
    )
    parser.add_argument(
        "--console-oauth",
        action="store_true",
        help="Use console-based OAuth instead of opening a browser window",
    )
    parser.add_argument(
        "--replace-events",
        action="store_true",
        help="Delete matching events on the target calendar before inserting new ones",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not call Google Calendar; just print extracted events")
    return parser


def main(args: list[str] | None = None) -> None:
    parser = build_parser()
    opts = parser.parse_args(args=args)

    try:
        default_start_hour, default_start_minute = _parse_time_arg(opts.default_start)
    except ValueError as exc:
        parser.error(str(exc))

    console = Console()
    try:
        report = load_schedule_csv(
            csv_path=opts.csv,
            timezone=opts.timezone,
            default_start_hour=default_start_hour,
            default_start_minute=default_start_minute,
            default_duration_minutes=opts.duration,
            fallback_year=opts.csv_year,
        )
    except Exception as exc:  # pragma: no cover - invalid file path
        console.print(f"[red]Failed to load CSV: {exc}[/]")
        return

    all_events: List[CourseEvent] = report.events

    if report.warnings:
        console.print("[yellow]Warnings during CSV load:[/]")
        for warning in report.warnings:
            console.print(f"- {warning}")

    if not all_events:
        console.print("[yellow]No events extracted.[/]")
        return

    all_events = all_events[: opts.max_events]
    _render_preview(console, all_events)

    client = GoogleCalendarClient(
        calendar_id=opts.calendar,
        credentials_file=opts.credentials,
        token_file=opts.token,
        dry_run=opts.dry_run,
        use_console_oauth=opts.console_oauth,
    )

    if opts.replace_events:
        if opts.dry_run:
            console.print("[yellow]Replace flag ignored during dry-run; no events deleted.[/]")
        else:
            deleted = client.delete_matching_events(all_events)
            console.print(f"[yellow]Deleted {deleted} existing events before inserting new ones.[/]")

    results = client.create_events(all_events)
    if opts.dry_run:
        console.print("[green]Dry run complete. Events were not created.[/]")
        for payload in results:
            console.print(payload)
    else:
        console.print(f"[green]Created {len(results)} Google Calendar events.[/]")


def _render_preview(console: Console, events: List[CourseEvent]) -> None:
    table = Table(title="Extracted Events")
    table.add_column("Title")
    table.add_column("Category")
    table.add_column("Start")
    table.add_column("End")
    for event in events:
        table.add_row(event.title, event.category.value, str(event.start), str(event.end or event.start))
    console.print(table)


def _parse_time_arg(value: str) -> tuple[int, int]:
    try:
        hour_str, minute_str = value.split(":")
        hour = int(hour_str)
        minute = int(minute_str)
    except ValueError as exc:
        raise ValueError("default start time must be formatted as HH:MM (24-hour)") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("default start time must use 0<=HH<=23 and 0<=MM<=59")
    return hour, minute


if __name__ == "__main__":  # pragma: no cover
    main()
