import json

from fastapi.testclient import TestClient

import nanoserve.server as server_module
from nanoserve.config import Settings
from nanoserve.engine import MockEngine
from nanoserve.server import app, state


def make_client() -> TestClient:
    state.settings = Settings(engine="mock", model_id="mock-27b")
    state.engine = MockEngine(model_id="mock-27b", words_per_second=100_000)
    state.last_stats = None
    return TestClient(app)


def test_health():
    c = make_client()
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["engine"] == "mock"
    assert body["model"] == "mock-27b"


def test_chat_completion_non_streaming():
    c = make_client()
    r = c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello there friend"}],
        "max_tokens": 8,
    })
    assert r.status_code == 200
    body = r.json()
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert len(choice["message"]["content"].split()) <= 8
    usage = body["usage"]
    assert usage["prompt_tokens"] == 3
    assert usage["completion_tokens"] == 8
    # serving metrics ride along for easy evals
    assert body["nanoserve_stats"]["decode_tps"] > 0


def test_chat_completion_streaming_sse():
    c = make_client()
    with c.stream("POST", "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
        "stream": True,
    }) as resp:
        events = []
        for line in resp.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append(json.loads(line[len("data: "):]))
    deltas = [e["choices"][0]["delta"]["content"]
              for e in events if not e["choices"][0]["finish_reason"]]
    assert len(deltas) >= 5
    assert "".join(deltas).strip()  # non-empty text arrived token by token
    finish = [e for e in events if e["choices"][0]["finish_reason"]]
    assert finish and finish[0]["choices"][0]["finish_reason"] == "stop"


def test_metrics_after_request():
    c = make_client()
    assert c.get("/metrics").json()["last_request"] is None
    c.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hey"}], "max_tokens": 4,
    })
    last = c.get("/metrics").json()["last_request"]
    assert last is not None
    assert last["engine"] == "mock"
    assert last["output_tokens"] == 4


def test_lazy_engine_load():
    state.settings = Settings(engine="mock", model_id="lazy-27b")
    state.engine = None
    try:
        client = TestClient(app)
        body = client.get("/health").json()
        assert body["engine_loaded"] is False
        client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "wake up"}], "max_tokens": 2,
        })
        assert state.engine is not None and state.engine.model_id == "lazy-27b"
    finally:
        state.engine = None
