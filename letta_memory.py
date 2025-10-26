"""Lightweight client that emulates a LeTTA-style memory service for feedback."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import datetime as dt


@dataclass
class PreferenceProfile:
    """Aggregated view of a user's expressed meal preferences."""

    item_scores: Dict[str, float]
    tag_scores: Dict[str, float]
    category_scores: Dict[str, float]
    location_scores: Dict[str, float]
    last_updated: Optional[str] = None

    def has_signal(self) -> bool:
        """Return True when any preference weights are non-zero."""
        return any(
            any(values.values())
            for values in (
                self.item_scores,
                self.tag_scores,
                self.category_scores,
                self.location_scores,
            )
        )


class LettaMemoryClient:
    """Stores up/down votes with simple heuristics to mimic LeTTA memory."""

    def __init__(self, storage_path: Optional[Path] = None) -> None:
        default_root = Path(os.getenv("LETTA_STORAGE_ROOT", ".cache")) / "letta"
        self.storage_path = storage_path or (default_root / "feedback.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record_feedback(
        self,
        user_id: str,
        item_id: str,
        *,
        tags: Sequence[str],
        category: Optional[str],
        location: Optional[str],
        vote: int,
    ) -> PreferenceProfile:
        """Persist a feedback event and return the updated profile."""
        if vote not in (-1, 1):
            raise ValueError("vote must be +1 (upvote) or -1 (downvote)")
        with self._lock:
            state = self._load_state()
            user = state.setdefault(
                user_id,
                {
                    "items": {},
                    "tags": {},
                    "categories": {},
                    "locations": {},
                    "history": [],
                    "last_updated": None,
                },
            )

            self._bump(user["items"], item_id, vote)
            for tag in tags:
                self._bump(user["tags"], tag.lower(), vote)
            if category:
                self._bump(user["categories"], category.lower(), vote)
            if location:
                self._bump(user["locations"], location.lower(), vote)

            timestamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
            user["history"].append({"item_id": item_id, "vote": vote, "at": timestamp})
            user["history"] = user["history"][-200:]  # avoid unbounded growth
            user["last_updated"] = timestamp

            self._save_state(state)
            return self._profile_from_state(user)

    def get_preference_profile(self, user_id: str) -> PreferenceProfile:
        """Fetch the stored profile for a user."""
        with self._lock:
            state = self._load_state()
            user = state.get(
                user_id,
                {
                    "items": {},
                    "tags": {},
                    "categories": {},
                    "locations": {},
                    "last_updated": None,
                },
            )
            return self._profile_from_state(user)

    def clear_feedback(self, user_id: str) -> None:
        """Remove all stored feedback for the given user."""
        with self._lock:
            state = self._load_state()
            if user_id in state:
                del state[user_id]
                self._save_state(state)

    def _load_state(self) -> Dict[str, Any]:
        if not self.storage_path.exists():
            return {}
        try:
            raw = self.storage_path.read_text(encoding="utf-8")
        except OSError:
            return {}
        try:
            state = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return state if isinstance(state, dict) else {}

    def _save_state(self, state: Dict[str, Any]) -> None:
        payload = json.dumps(state, indent=2, sort_keys=True)
        tmp_path = self.storage_path.with_suffix(".tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(self.storage_path)

    @staticmethod
    def _bump(bucket: Dict[str, float], key: str, delta: float) -> None:
        new_value = bucket.get(key, 0.0) + float(delta)
        if abs(new_value) < 1e-6:
            bucket.pop(key, None)
        else:
            bucket[key] = new_value

    @staticmethod
    def _profile_from_state(state: Dict[str, Any]) -> PreferenceProfile:
        return PreferenceProfile(
            item_scores=dict(state.get("items", {})),
            tag_scores=dict(state.get("tags", {})),
            category_scores=dict(state.get("categories", {})),
            location_scores=dict(state.get("locations", {})),
            last_updated=state.get("last_updated"),
        )

