"""A minimal, mostly-OpenAI-compatible chat server.

Endpoints:
    GET  /health               -> liveness + which engine/model is loaded
    GET  /metrics              -> stats (ttft, tok/s) from the last request
    POST /v1/chat/completions  -> chat completion, streaming or not

Run:
    NANOSERVE_ENGINE=mock python -m nanoserve.server          # no weights needed
    NANOSERVE_ENGINE=mlx NANOSERVE_MODEL=<repo> python -m nanoserve.server
    NANOSERVE_ENGINE=nanotrain NANOSERVE_MODEL=<checkpoint> python -m nanoserve.server

The response shapes mirror the OpenAI API closely enough that most eval and
client tooling works unchanged; this is deliberately ~150 lines, not a
production gateway.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .config import Settings, load_settings
from .engine import BaseEngine, Message, build_engine


class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str | None = None          # accepted for OpenAI compat; engine owns the model
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=None)
    temperature: float | None = None
    stream: bool = False


class NanoServeState:
    """Holds the loaded engine + stats from the most recent generation."""

    engine: BaseEngine | None = None
    last_stats: dict | None = None
    settings: Settings


state = NanoServeState()


def get_engine() -> BaseEngine:
    if state.engine is None:  # lazy-load so `mock` mode starts instantly
        s = state.settings
        state.engine = build_engine(
            s.resolved_engine(), s.model_id, thinking=s.thinking
        )
    return state.engine


app = FastAPI(title="nanoserve")


@app.get("/health")
def health() -> dict:
    engine_loaded = state.engine is not None
    return {
        "status": "ok",
        "engine": state.settings.resolved_engine(),
        "model": state.settings.model_id,
        "engine_loaded": engine_loaded,
    }


@app.get("/metrics")
def metrics() -> dict:
    return {"last_request": state.last_stats}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    engine = get_engine()
    messages: list[Message] = [m.model_dump() for m in req.messages]
    max_tokens = req.max_tokens if req.max_tokens is not None else state.settings.max_tokens
    temperature = req.temperature if req.temperature is not None else state.settings.temperature
    created = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if not req.stream:
        result = engine.generate(messages, max_tokens=max_tokens, temperature=temperature)
        state.last_stats = result.stats.to_dict()
        return _completion_payload(completion_id, created, result.text, result.stats.to_dict())

    def sse() -> Iterator[str]:
        for delta, stats in engine.stream_stats(
            messages, max_tokens=max_tokens, temperature=temperature
        ):
            if stats is not None:
                state.last_stats = stats.to_dict()
                yield f"data: {_sse_chunk(completion_id, created, '', finish=True)}\n\n"
                yield "data: [DONE]\n\n"
                break
            yield f"data: {_sse_chunk(completion_id, created, delta)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


# --------------------------------------------------------------------------- #
# Response shaping
# --------------------------------------------------------------------------- #


def _choice_delta(content: str, finish: bool) -> dict:
    message = {"role": "assistant", "content": content}
    return {
        "index": 0,
        "delta": message,
        "finish_reason": "stop" if finish else None,
    }


def _sse_chunk(cid: str, created: int, content: str, finish: bool = False) -> str:
    return json.dumps({
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": state.settings.model_id,
        "choices": [_choice_delta(content, finish)],
    })


def _completion_payload(cid: str, created: int, text: str, stats: dict) -> dict:
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": state.settings.model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": stats["prompt_tokens"],
            "completion_tokens": stats["output_tokens"],
            "total_tokens": stats["prompt_tokens"] + stats["output_tokens"],
        },
        "nanoserve_stats": stats,   # extension: serving metrics inline
    }


def main() -> None:
    import uvicorn

    state.settings = load_settings()
    uvicorn.run(app, host=state.settings.host, port=state.settings.port, log_level="info")


if __name__ == "__main__":
    main()
