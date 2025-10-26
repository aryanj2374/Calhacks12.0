from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Set

from rich.console import Console
from rich.table import Table

from .calendar_client import GoogleCalendarClient
from .email_event_extractor import EmailEventExtractor
from .gmail_client import GmailClient, GmailMessage
from .llm_client import LavaPaymentsLLMClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch Gmail messages and turn them into calendar events.")
    parser.add_argument("--calendar", required=True, help="Google Calendar ID to insert events into")
    parser.add_argument("--gmail-credentials", default="credentials.json", help="Path to OAuth client credentials")
    parser.add_argument("--gmail-token", default="gmail_token.json", help="Path to cached Gmail OAuth token")
    parser.add_argument("--calendar-credentials", default="credentials.json", help="OAuth credentials for Calendar use")
    parser.add_argument("--calendar-token", default="token.json", help="Cached OAuth token for Calendar writes")
    parser.add_argument(
        "--query",
        default="label:unread newer_than:1d",
        help="Gmail search query to locate relevant emails (default filters to unread from the last 24 hours)",
    )
    parser.add_argument("--timezone", default="America/Los_Angeles", help="Timezone hint for event extraction")
    parser.add_argument("--forward-token", default=None, help="Explicit Lava forward token (falls back to env if missing)")
    parser.add_argument("--provider", default="openai", help="LLM provider hint passed to Lava")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model requested from Lava")
    parser.add_argument("--apply", action="store_true", help="Create Google Calendar events (otherwise just preview)")
    parser.add_argument(
        "--processed-store",
        default=".gmail_processed.json",
        help="Path to a JSON file tracking processed Gmail message IDs",
    )
    parser.add_argument(
        "--console-oauth",
        action="store_true",
        help="Use copy/paste OAuth flow (no automatic browser launch)",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    parser = build_parser()
    opts = parser.parse_args(args=args)
    console = Console()

    gmail_client = GmailClient(
        credentials_file=opts.gmail_credentials,
        token_file=opts.gmail_token,
        use_console_oauth=opts.console_oauth,
    )
    try:
        raw_messages = gmail_client.fetch_messages(query=opts.query)
    except Exception as exc:
        console.print(f"[red]Failed to fetch Gmail messages: {exc}[/]")
        return

    store = ProcessedMessageStore(Path(opts.processed_store))
    new_messages = store.filter_new(raw_messages)
    if not new_messages:
        console.print("[yellow]No new Gmail messages found (or all were previously processed).[/]")
        return

    forward_token = opts.forward_token or None
    llm_client = LavaPaymentsLLMClient(
        forward_token=forward_token,
        provider=opts.provider,
        model=opts.model,
    )
    extractor = EmailEventExtractor(
        llm_client,
        timezone_name=opts.timezone,
    )

    events = []
    for message in new_messages:
        try:
            events.extend(extractor.extract_events(message))
        except Exception as exc:
            console.print(f"[red]Failed to parse message '{message.subject}': {exc}[/]")

    if not events:
        console.print("[yellow]No events detected in the new Gmail messages.[/]")
        return

    _render_events(console, events)

    if not opts.apply:
        console.print("[cyan]Preview complete. Rerun with --apply to create these events.[/]")
        return

    calendar_client = GoogleCalendarClient(
        calendar_id=opts.calendar,
        credentials_file=opts.calendar_credentials,
        token_file=opts.calendar_token,
        dry_run=False,
        use_console_oauth=opts.console_oauth,
    )
    created_ids = calendar_client.create_events(events)
    console.print(f"[green]Created {len(created_ids)} Google Calendar events.[/]")
    store.mark_processed(new_messages)


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
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self.ids = set(str(item) for item in data)
        except json.JSONDecodeError:
            self.ids = set()

    def filter_new(self, messages: Iterable[GmailMessage]) -> List[GmailMessage]:
        return [message for message in messages if message.id not in self.ids]

    def mark_processed(self, messages: Iterable[GmailMessage]) -> None:
        for message in messages:
            self.ids.add(message.id)
        self.path.write_text(json.dumps(sorted(self.ids)), encoding="utf-8")


def _render_events(console: Console, events) -> None:
    table = Table(title="Events from Gmail")
    table.add_column("Title")
    table.add_column("Category")
    table.add_column("Start")
    table.add_column("End")
    table.add_column("Source")
    for event in events:
        table.add_row(
            event.title,
            event.category.value,
            event.start.isoformat(),
            (event.end or event.start).isoformat(),
            event.source_url or "",
        )
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    main()
