"""Provider-neutral chat client, spoken over the OpenAI chat-completions wire
format so any OpenRouter model can drive the harness.

Deliberately ~1 file, no SDK: the entire request/response path is visible, and
swapping ``anthropic/claude-sonnet-4.5`` for ``openai/gpt-4.1`` or a local model
is one env var. Weaker models degrade the recovery loop but the contract is the
same.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


def load_dotenv(path: str | os.PathLike = "") -> None:
    """Populate os.environ from a .env file (no dependency). Walks up from the
    package to find one if no path is given. Existing env vars win."""
    from pathlib import Path

    if path:
        candidates = [Path(path)]
    else:
        here = Path(__file__).resolve()
        candidates = [p / ".env" for p in (here.parent, *here.parents[:3])]
    for env_path in candidates:
        if not env_path.is_file():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)
        return


load_dotenv()

DEFAULT_MODEL = os.environ.get("AGENTICFOAM_MODEL", "nvidia/nemotron-3.5-lightning:free")
DEFAULT_BASE_URL = os.environ.get("AGENTICFOAM_BASE_URL", "https://openrouter.ai/api/v1")


class LLMError(RuntimeError):
    pass


@dataclass
class ToolCallReq:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCallReq]
    finish_reason: str
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._key = api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self._client = httpx.Client(timeout=timeout)
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    # -- public ----------------------------------------------------------------
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str = "auto",
    ) -> Reply:
        if not self._key:
            raise LLMError(
                "no API key: set OPENROUTER_API_KEY (or OPENAI_API_KEY) in the environment"
            )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice

        data = self._post(body)
        choice = data["choices"][0]
        msg = choice.get("message", {})
        usage = data.get("usage", {}) or {}
        self.calls += 1
        self.tokens_in += usage.get("prompt_tokens", 0)
        self.tokens_out += usage.get("completion_tokens", 0)

        tcs: list[ToolCallReq] = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError:
                args = {"__unparsed__": raw}
            tcs.append(ToolCallReq(id=tc.get("id", f"call_{len(tcs)}"), name=fn.get("name", ""), arguments=args))

        return Reply(
            text=msg.get("content") or "",
            tool_calls=tcs,
            finish_reason=choice.get("finish_reason", ""),
            usage=usage,
        )

    # -- internals -----------------------------------------------------------
    def _post(self, body: dict, retries: int = 4) -> dict:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
            "X-Title": "agenticFOAM",
        }
        last_err: Exception | None = None
        for attempt in range(retries):
            try:
                resp = self._client.post(url, headers=headers, json=body)
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError("retryable", request=resp.request, response=resp)
                resp.raise_for_status()
                return resp.json()
            except (httpx.HTTPStatusError, httpx.TransportError) as e:
                last_err = e
                if attempt == retries - 1:
                    break
                time.sleep(1.5 * (2 ** attempt))
        raise LLMError(f"OpenRouter request failed after {retries} attempts: {last_err}")


# -- message helpers ------------------------------------------------------------
def sys_msg(text: str) -> dict:
    return {"role": "system", "content": text}


def user_msg(text: str) -> dict:
    return {"role": "user", "content": text}


def assistant_msg(reply: Reply) -> dict:
    """Reconstruct the assistant turn (with tool_calls) for the next request."""
    m: dict[str, Any] = {"role": "assistant", "content": reply.text or None}
    if reply.tool_calls:
        m["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in reply.tool_calls
        ]
    return m


def extract_json(text: str, default: dict | None = None) -> dict:
    """First balanced-ish {...} object in a model reply, parsed. Falls back to
    ``default`` (or {}) on anything unparseable -- callers decide what a missing
    field means."""
    import re

    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return dict(default or {})
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return dict(default or {})


def tool_msg(call_id: str, result: Any) -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": result if isinstance(result, str) else json.dumps(result, default=str),
    }
