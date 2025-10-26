"""Chat agent that combines the menu recommender with an Ollama-hosted LLM."""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from menu_recommender import MenuRecommender, build_recommender


DEFAULT_MODEL = "llama3.1"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


@dataclass
class ChatResult:
    """Container for chat responses."""

    response: str
    used_llm: bool
    recommendations: List[dict[str, Any]]
    memory_used: bool = False
    memory_context: Optional[Dict[str, Any]] = None


class OllamaClient:
    """Minimal Ollama chat client."""

    def __init__(self, model: str, host: str = DEFAULT_OLLAMA_HOST) -> None:
        self.model = model
        self.host = host.rstrip("/")

    def chat(self, messages: List[dict[str, str]], stream: bool = False) -> str:
        try:
            return self._chat_api(messages, stream=stream)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                try:
                    return self._generate_api(messages)
                except requests.HTTPError as gen_exc:
                    if gen_exc.response is not None and gen_exc.response.status_code == 404:
                        return self._cli_generate(messages)
                    raise
            raise
        except (requests.ConnectionError, requests.Timeout):
            return self._cli_generate(messages)

    def _build_prompt_text(self, messages: List[dict[str, str]]) -> str:
        prompt_parts = []
        for message in messages:
            role = message.get("role", "user").upper()
            content = message.get("content", "")
            prompt_parts.append(f"{role}:\n{content}")
        return "\n\n".join(prompt_parts) + "\nASSISTANT:\n"

    def _cli_generate(self, messages: List[dict[str, str]]) -> str:
        prompt = self._build_prompt_text(messages)
        env = os.environ.copy()
        cmd = ["ollama", "run", self.model, prompt]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ollama CLI not found on PATH.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"ollama CLI returned an error: {exc.stderr.strip()}") from exc

        output = result.stdout.strip()
        if not output:
            raise RuntimeError("No content returned from ollama CLI.")
        return output

    def _chat_api(self, messages: List[dict[str, str]], stream: bool = False) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        response = requests.post(
            f"{self.host}/api/chat",
            json=payload,
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "60")),
        )
        response.raise_for_status()
        data = response.json()
        if stream:
            # When streaming, Ollama returns chunks; ensure we collect content.
            content = "".join(
                chunk.get("message", {}).get("content", "")
                for chunk in data
                if isinstance(chunk, dict)
            )
        else:
            content = data.get("message", {}).get("content", "")
        if not content:
            raise RuntimeError("No response content returned from Ollama.")
        return content.strip()

    def _generate_api(self, messages: List[dict[str, str]]) -> str:
        prompt = self._build_prompt_text(messages)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        response = requests.post(
            f"{self.host}/api/generate",
            json=payload,
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "60")),
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("response", "")
        if not content:
            raise RuntimeError("No response content returned from Ollama /api/generate.")
        return content.strip()


class DiningChatAgent:
    """High-level chat interface that grounds answers in the menu recommender."""

    def __init__(
        self,
        recommender: MenuRecommender,
        model: str = DEFAULT_MODEL,
        client: Optional[Any] = None,
    ) -> None:
        self.recommender = recommender
        self.model = model
        self.client = client or self._init_default_client()

    def _init_default_client(self) -> Optional[Any]:
        """Create an Ollama client if the server is accessible."""
        host = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        return OllamaClient(model=self.model, host=host)

    def respond(self, query: str, top_k: int = 5, user_id: Optional[str] = None) -> ChatResult:
        """Return a response grounded in the recommender results."""
        recommendations = self.recommender.recommend(query, top_k=top_k, user_id=user_id)
        memory_context = self.recommender.last_memory_context

        if not recommendations:
            return ChatResult(
                response="I could not find any dishes that match that request. Try rephrasing or widening your preferences.",
                used_llm=False,
                recommendations=[],
                memory_used=bool(memory_context.get("applied")),
                memory_context=memory_context,
            )

        prompt = self._build_prompt(query, recommendations, memory_context)

        if self.client is None:
            return ChatResult(
                response=self._fallback_response(recommendations),
                used_llm=False,
                recommendations=recommendations,
                memory_used=bool(memory_context.get("applied")),
                memory_context=memory_context,
            )

        try:
            message = self.client.chat(prompt)
        except Exception as exc:  # noqa: BLE001
            return ChatResult(
                response=(
                    "I had trouble contacting the Ollama service. "
                    "Here are matches directly from the menu data:\n"
                    f"{self._format_recommendations(recommendations, include_header=False)}\n\n"
                    f"(Error: {exc})"
                ),
                used_llm=False,
                recommendations=recommendations,
                memory_used=bool(memory_context.get("applied")),
                memory_context=memory_context,
            )

        return ChatResult(
            response=message,
            used_llm=True,
            recommendations=recommendations,
            memory_used=bool(memory_context.get("applied")),
            memory_context=memory_context,
        )

    def record_feedback(self, user_id: str, item_id: str, vote: int) -> Dict[str, Any]:
        """Persist an upvote/downvote for a recommendation."""
        profile = self.recommender.record_feedback(user_id, item_id, vote)
        return {
            "user_id": user_id,
            "item_id": item_id,
            "vote": vote,
            "memory_used": profile.has_signal(),
            "profile": {
                "items": profile.item_scores,
                "tags": profile.tag_scores,
                "categories": profile.category_scores,
                "locations": profile.location_scores,
            },
            "last_updated": profile.last_updated,
        }

    def _build_prompt(
        self,
        query: str,
        recommendations: List[dict[str, Any]],
        memory_context: Optional[Dict[str, Any]],
    ) -> List[dict[str, str]]:
        """Construct LLM-ready prompt messages."""
        context_lines = []
        for idx, item in enumerate(recommendations, start=1):
            nutrition = item.get("nutrition", {})
            nutrition_line = ", ".join(f"{k}: {v}" for k, v in nutrition.items())
            classification = item.get("classification_hints", {})
            tag_line = ""
            tags = classification.get("tags") or []
            if tags:
                tag_line = f" tags: {', '.join(tags)};"
            category_hint = classification.get("category") or item.get("category")
            context_lines.append(
                (
                    f"{idx}. {item['name']} — location: {item['location']} ({item['meal']}), "
                    f"category: {category_hint}; dietary: {', '.join(item['dietary_choices']) or 'None'};"
                    f"{tag_line} "
                    f"nutrition: {nutrition_line or 'N/A'}; ingredients: {item.get('ingredients') or 'N/A'}."
                ).strip()
            )

        context_blob = "\n".join(context_lines)
        preference_summary = self._summarize_preferences(memory_context)
        preference_block = f"Known user preferences:\n{preference_summary}\n\n" if preference_summary else ""

        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful dining concierge for UC Berkeley students. "
                    "Always ground your answers in the supplied menu data. "
                    "List the top five dishes with short descriptions, highlight why they fit the user request, "
                    "and mention the dining hall and meal period."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User request: {query}\n\n"
                    f"{preference_block}"
                    f"Top matches from the dining menus:\n{context_blob}\n\n"
                    "Compose a concise recommendation list of up to five items using this data."
                ),
            },
        ]

    def _summarize_preferences(self, memory_context: Optional[Dict[str, Any]]) -> str:
        if not memory_context or not memory_context.get("memory_used"):
            return ""
        profile = memory_context.get("profile") or {}
        lines: List[str] = []

        def top_entries(entries: Dict[str, float], positive: bool = True) -> List[tuple[str, float]]:
            filtered = [
                (key, score)
                for key, score in entries.items()
                if (score > 0 and positive) or (score < 0 and not positive)
            ]
            return sorted(filtered, key=lambda pair: abs(pair[1]), reverse=True)[:3]

        item_scores = profile.get("items", {})
        liked_items = top_entries(item_scores, positive=True)
        disliked_items = top_entries(item_scores, positive=False)
        if liked_items:
            parts = [
                f"{self._resolve_item_name(item_id)} (+{score:.0f})"
                for item_id, score in liked_items
            ]
            lines.append(f"Enjoyed dishes: {', '.join(parts)}.")
        if disliked_items:
            parts = [
                f"{self._resolve_item_name(item_id)} ({score:.0f})"
                for item_id, score in disliked_items
            ]
            lines.append(f"Disliked dishes: {', '.join(parts)}.")

        tag_scores = profile.get("tags", {})
        fav_tags = top_entries(tag_scores, positive=True)
        if fav_tags:
            lines.append(
                "Positive signals for tags: "
                + ", ".join(f"{tag} (+{score:.0f})" for tag, score in fav_tags)
                + "."
            )

        category_scores = profile.get("categories", {})
        fav_categories = top_entries(category_scores, positive=True)
        if fav_categories:
            lines.append(
                "Preferred categories: "
                + ", ".join(f"{category} (+{score:.0f})" for category, score in fav_categories)
                + "."
            )

        location_scores = profile.get("locations", {})
        fav_locations = top_entries(location_scores, positive=True)
        if fav_locations:
            lines.append(
                "Preferred locations: "
                + ", ".join(f"{location.title()} (+{score:.0f})" for location, score in fav_locations)
                + "."
            )

        return "\n".join(lines)

    def _resolve_item_name(self, item_id: str) -> str:
        item = self.recommender.kb.item_lookup.get(item_id)
        return item.name if item else item_id

    @staticmethod
    def _fallback_response(recommendations: List[dict[str, Any]]) -> str:
        return DiningChatAgent._format_recommendations(recommendations, include_header=True)

    @staticmethod
    def _format_recommendations(
        recommendations: List[dict[str, Any]], include_header: bool = True
    ) -> str:
        lines: List[str] = []
        if include_header:
            lines.append("LLM backend unavailable. Here are the top matches based on menu data:")
        for idx, item in enumerate(recommendations, start=1):
            nutrition = item.get("nutrition", {})
            nutrition_line = ", ".join(f"{k}: {v}" for k, v in nutrition.items())
            description = item.get("blurb") or item.get("context", "")
            lines.append(
                f"{idx}. {item['name']} — {description} (Nutrition: {nutrition_line or 'N/A'})"
            )
        return "\n".join(lines)


def load_agent(json_path: Path | str, model: str = DEFAULT_MODEL) -> DiningChatAgent:
    path = Path(json_path)
    recommender = build_recommender(path)
    return DiningChatAgent(recommender, model=model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the Berkeley dining assistant.")
    parser.add_argument("query", nargs="?", help="Single-turn question to ask the assistant.")
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("menus.json"),
        help="Path to the menus JSON file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model identifier (default: llama3.1).",
    )
    parser.add_argument(
        "--user",
        help="Optional user identifier for personalized recommendations.",
    )
    args = parser.parse_args()

    agent = load_agent(args.json, model=args.model)
    user_id = args.user

    if args.query:
        result = agent.respond(args.query, user_id=user_id)
        print(result.response)
        return

    print("Interactive dining assistant. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        result = agent.respond(user_input, user_id=user_id)
        print(result.response)


if __name__ == "__main__":
    main()
