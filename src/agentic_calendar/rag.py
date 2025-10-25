from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Sequence

from .models import CourseEvent

TOKEN_RE = re.compile(r"\w+")


@dataclass
class RAGChunk:
    """Lightweight representation of an event snippet for retrieval."""

    event: CourseEvent
    text: str


class CalendarRAGIndex:
    """
    Extremely small-footprint retrieval module.

    Instead of relying on heavy embeddings, it tokenizes each event title/description
    and calculates overlap scores against the user's request. While simple, this keeps
    the dependency footprint low and still surfaces the most relevant calendar entries
    for context-grounded prompting.
    """

    def __init__(self, events: Sequence[CourseEvent]):
        self.chunks: List[RAGChunk] = [
            RAGChunk(
                event=event,
                text=self._format_event_text(event),
            )
            for event in events
        ]

    def build_context(self, question: str, top_k: int = 5) -> str:
        """Return a plain-text context block describing the most relevant events."""
        if not self.chunks:
            return "No events currently scheduled."

        scored = self._score_chunks(question)
        top = scored[: max(1, top_k)]
        lines = []
        for rank, chunk in enumerate(top, start=1):
            event = chunk.event
            end_iso = (event.end or event.start).isoformat()
            lines.append(
                f"{rank}. {event.title} [{event.category.value}] "
                f"{event.start.isoformat()} → {end_iso} @ {event.location or 'TBD'} :: {event.description or ''}".strip()
            )
        return "\n".join(lines)

    def _score_chunks(self, question: str) -> List[RAGChunk]:
        query_tokens = self._tokenize(question)
        if not query_tokens:
            return self.chunks
        scored = []
        now = datetime.now(timezone.utc)
        for chunk in self.chunks:
            event_start = chunk.event.start
            if event_start.tzinfo is None:
                event_start = event_start.replace(tzinfo=timezone.utc)
            chunk_tokens = self._tokenize(chunk.text)
            overlap = len(query_tokens & chunk_tokens)
            delta_days = abs((event_start - now).days)
            freshness = 1.0 / (1 + math.log1p(delta_days))
            # Tiny freshness term keeps ordering deterministic even when overlap=0.
            score = overlap + freshness
            scored.append((score, chunk))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [chunk for _, chunk in scored]

    @staticmethod
    def _format_event_text(event: CourseEvent) -> str:
        parts = [
            event.title,
            event.category.value,
            event.start.isoformat(),
        ]
        if event.description:
            parts.append(event.description)
        if event.location:
            parts.append(event.location)
        return " ".join(parts)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(TOKEN_RE.findall(text.lower()))
