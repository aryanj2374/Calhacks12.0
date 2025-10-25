from __future__ import annotations

import json
import os
from typing import Iterable, List, Mapping, MutableMapping, Optional
from urllib.parse import quote_plus

import requests


class LLMError(RuntimeError):
    """Raised when the LavaPayments API returns an error payload."""


class LavaPaymentsLLMClient:
    """
    Minimal client for LavaPayments' forward proxy.

    Lava routes requests through `/v1/forward?u=<provider_endpoint>` and expects a
    *forward token* (not a secret key) in the `Authorization: Bearer <token>` header.
    Configuration is pulled from environment variables unless provided directly:

    - ``LAVAPAY_FORWARD_TOKEN`` / ``LAVA_FORWARD_TOKEN`` – forward token JSON string
    - ``LAVAPAY_BASE_URL`` – defaults to ``https://api.lavapayments.com/v1``
    - ``LAVAPAY_TARGET_URL`` – upstream provider URL (defaults to OpenAI chat completions)
    - ``LAVAPAY_MODEL`` – provider-specific model id (``gpt-4o-mini`` etc.)
    """

    def __init__(
        self,
        forward_token: str | None = None,
        *,
        api_key: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        target_url: str | None = None,
        timeout: int = 30,
        temperature: float = 0.2,
    ):
        token = forward_token or api_key
        if not token:
            token = (
                os.environ.get("LAVAPAY_FORWARD_TOKEN")
                or os.environ.get("LAVA_FORWARD_TOKEN")
                or os.environ.get("LAVAPAY_API_KEY")
                or os.environ.get("LAVA_API_KEY")
            )
        if not token:
            raise ValueError(
                "Lava forward token missing. Copy the 'Self Forward Token' from Build > Secret Keys "
                "and set LAVAPAY_FORWARD_TOKEN or pass --forward-token."
            )
        if token.startswith("lava_sk"):
            raise ValueError(
                "It looks like you provided a Lava secret key (starts with 'lava_sk'). "
                "The forward proxy requires a forward token JSON payload. Copy the "
                "'Self Forward Token' from Build > Secret Keys and use that instead."
            )

        self.forward_token = token
        self.provider = provider or os.environ.get("LAVAPAY_PROVIDER", "openai")
        self.model = model or os.environ.get("LAVAPAY_MODEL", "gpt-4o-mini")
        self.base_url = base_url or os.environ.get("LAVAPAY_BASE_URL", "https://api.lavapayments.com/v1")
        self.target_url = target_url or os.environ.get("LAVAPAY_TARGET_URL") or self._default_target_url(self.provider)
        self.timeout = timeout
        self.temperature = temperature

    def chat(
        self,
        messages: Iterable[Mapping[str, str]],
        *,
        system_prompt: str | None = None,
        response_format: Optional[MutableMapping[str, str]] = None,
    ) -> str:
        """
        Call LavaPayments' chat completions endpoint and return the assistant text response.

        ``messages`` should already include the previous conversation turns.
        ``system_prompt`` is prepended automatically if provided.
        """

        payload: dict[str, object] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": self.temperature,
        }
        if system_prompt:
            payload["messages"] = [{"role": "system", "content": system_prompt}, *payload["messages"]]
        if response_format:
            payload["response_format"] = response_format

        forward_endpoint = f"{self.base_url.rstrip('/')}/forward?u={quote_plus(self.target_url)}"
        headers = {
            "Authorization": f"Bearer {self.forward_token}",
            "Content-Type": "application/json",
        }
        response = requests.post(forward_endpoint, headers=headers, json=payload, timeout=self.timeout)
        if response.status_code >= 400:
            raise LLMError(f"LavaPayments API error {response.status_code}: {response.text}")

        try:
            body = response.json()
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
            raise LLMError(f"Invalid JSON from LavaPayments: {response.text[:200]}") from exc

        choices: List[Mapping[str, object]] = body.get("choices", [])
        if not choices:
            raise LLMError(f"LavaPayments response missing choices: {body}")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMError(f"LavaPayments response missing assistant text: {message}")
        return content

    @staticmethod
    def _default_target_url(provider: str) -> str:
        if provider.lower() == "openai":
            return "https://api.openai.com/v1/chat/completions"
        raise ValueError(
            f"Unknown provider '{provider}'. Pass target_url explicitly via constructor or LAVAPAY_TARGET_URL."
        )
