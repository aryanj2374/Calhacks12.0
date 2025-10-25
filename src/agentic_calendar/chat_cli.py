from __future__ import annotations

import argparse

from rich.console import Console

from .agent import CalendarAgent
from .calendar_client import GoogleCalendarClient
from .csv_loader import load_schedule_csv
from .llm_client import LavaPaymentsLLMClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Interactive calendar chat powered by LavaPayments LLMs")
    parser.add_argument("--csv", default="schedule.csv", help="Seed calendar context from this CSV file")
    parser.add_argument("--calendar", required=True, help="Google Calendar ID (or primary)")
    parser.add_argument("--credentials", default="credentials.json", help="Path to OAuth client credentials")
    parser.add_argument("--token", default="token.json", help="Path to cached OAuth token")
    parser.add_argument(
        "--forward-token",
        default=None,
        help="Lava forward token JSON (defaults to LAVAPAY_FORWARD_TOKEN env var)",
    )
    parser.add_argument("--api-key", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--provider", default="openai", help="LavaPayments provider (openai, anthropic, google, etc.)")
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name to request from LavaPayments")
    parser.add_argument("--base-url", default=None, help="Override LavaPayments base URL")
    parser.add_argument("--dry-run", action="store_true", help="Skip actual Google Calendar writes")
    parser.add_argument(
        "--timezone",
        default="America/Los_Angeles",
        help="Timezone used when parsing CSV events (only impacts seed data)",
    )
    parser.add_argument(
        "--default-start",
        default="09:00",
        help="Fallback start time used when CSV rows omit explicit times",
    )
    parser.add_argument("--duration", type=int, default=50, help="Default duration for CSV events without end times")
    return parser


def parse_time_arg(value: str) -> tuple[int, int]:
    hour_str, minute_str = value.split(":")
    hour = int(hour_str)
    minute = int(minute_str)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("default start time must use HH:MM in 24-hour time")
    return hour, minute


def main(args: list[str] | None = None) -> None:
    parser = build_parser()
    opts = parser.parse_args(args=args)
    console = Console()
    try:
        default_start_hour, default_start_minute = parse_time_arg(opts.default_start)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        report = load_schedule_csv(
            csv_path=opts.csv,
            timezone=opts.timezone,
            default_start_hour=default_start_hour,
            default_start_minute=default_start_minute,
            default_duration_minutes=opts.duration,
        )
    except Exception as exc:
        console.print(f"[red]Failed to load CSV context: {exc}[/]")
        return

    if not report.events:
        console.print("[yellow]No events found in CSV. The agent will have limited context.[/]")

    calendar_client = GoogleCalendarClient(
        calendar_id=opts.calendar,
        credentials_file=opts.credentials,
        token_file=opts.token,
        dry_run=opts.dry_run,
    )
    forward_token = opts.forward_token or opts.api_key
    llm_client = LavaPaymentsLLMClient(
        forward_token=forward_token,
        model=opts.model,
        provider=opts.provider,
        base_url=opts.base_url,
    )
    agent = CalendarAgent(events=report.events, calendar_client=calendar_client, llm_client=llm_client)

    console.print("[green]Calendar chat ready. Type 'exit' or 'quit' to stop.[/]")
    while True:
        try:
            user_text = input("You> ").strip()
        except EOFError:
            console.print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break
        try:
            result = agent.handle_request(user_text, dry_run=opts.dry_run)
        except Exception as exc:
            console.print(f"[red]Agent error: {exc}[/]")
            continue
        console.print(f"[cyan]{result.action.value}[/]: {result.detail}")


if __name__ == "__main__":  # pragma: no cover
    main()
