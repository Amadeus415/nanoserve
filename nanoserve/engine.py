"""Generation engines.

An Engine turns a chat into a stream of text chunks. Everything else in
nanoserve (server, benchmarks, plots) sits on this tiny interface, so you can
swap backends without touching anything downstream:

    MockEngine  - deterministic fake tokens, zero deps. Runs on any laptop;
                  used for tests and for exercising server/bench plumbing.
    MLXEngine   - real inference on Apple Silicon via mlx-lm.

Timing (TTFT, prefill/decode tok/s) is measured once, generically, in
`BaseEngine.stream_stats` — engines only produce tokens.
"""

from __future__ import annotations

import itertools
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


Message = dict  # {"role": "user"|"assistant"|"system", "content": str}


@dataclass
class Stats:
    """Measurements from a single generation. The raw material for evals."""

    engine: str
    model: str
    prompt_tokens: int
    output_tokens: int
    ttft_s: float                 # time to first token = prompt processing time
    total_s: float

    @property
    def prefill_tps(self) -> float:
        """Prompt tokens processed per second during prefill."""
        return self.prompt_tokens / self.ttft_s if self.ttft_s > 0 else 0.0

    @property
    def decode_tps(self) -> float:
        """Generated tokens/sec after the first one."""
        gen_time = self.total_s - self.ttft_s
        n = max(self.output_tokens - 1, 1)
        return n / gen_time if gen_time > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "ttft_s": round(self.ttft_s, 4),
            "total_s": round(self.total_s, 4),
            "prefill_tps": round(self.prefill_tps, 2),
            "decode_tps": round(self.decode_tps, 2),
        }


@dataclass
class GenerationResult:
    text: str
    stats: Stats


class BaseEngine(ABC):
    name: str = "base"
    model_id: str = "unknown"

    @abstractmethod
    def _stream(self, messages: list[Message], *, max_tokens: int, temperature: float) -> Iterator[str]:
        """Yield text deltas. Subclasses implement only this."""

    def count_tokens(self, messages: list[Message]) -> int:
        """Rough token estimate; MLX engine overrides with the real tokenizer."""
        chars = sum(len(m["content"]) for m in messages)
        return max(chars // 4, 1)

    def stream_stats(
        self, messages: list[Message], *, max_tokens: int, temperature: float
    ) -> Iterator[tuple[str, Stats | None]]:
        """Yield (delta, None) per token, then ('', Stats) at the end."""
        start = time.perf_counter()
        ttft = None
        pieces: list[str] = []
        for delta in self._stream(messages, max_tokens=max_tokens, temperature=temperature):
            if ttft is None:
                ttft = time.perf_counter() - start
            pieces.append(delta)
            yield delta, None
        total = time.perf_counter() - start
        stats = Stats(
            engine=self.name,
            model=self.model_id,
            prompt_tokens=self.count_tokens(messages),
            output_tokens=len(pieces),
            ttft_s=ttft if ttft is not None else total,
            total_s=total,
        )
        yield "", stats

    def generate(self, messages: list[Message], *, max_tokens: int, temperature: float) -> GenerationResult:
        """Non-streaming convenience wrapper that also returns stats."""
        pieces: list[str] = []
        stats = None
        for delta, s in self.stream_stats(messages, max_tokens=max_tokens, temperature=temperature):
            if s is not None:
                stats = s
            elif delta:
                pieces.append(delta)
        assert stats is not None, "engine produced no final stats"
        return GenerationResult(text="".join(pieces), stats=stats)


# --------------------------------------------------------------------------- #
# Mock engine: deterministic filler text, optional artificial latency so you
# can simulate slow/fast decode without a GPU.
# --------------------------------------------------------------------------- #

_MOCK_WORDS = (
    "the model serving loop is simple prefill then decode "
    "tokens stream out one at a time from the kv cache "
    "latency is dominated by memory bandwidth not compute "
).split()


class MockEngine(BaseEngine):
    name = "mock"

    def __init__(self, model_id: str = "mock-27b", words_per_second: float = 50.0):
        self.model_id = model_id
        self.delay = 1.0 / words_per_second if words_per_second > 0 else 0.0

    def count_tokens(self, messages: list[Message]) -> int:
        return sum(len(m["content"].split()) for m in messages)

    def _stream(self, messages, *, max_tokens, temperature):
        counter = itertools.count()
        for i in range(max_tokens):
            word = _MOCK_WORDS[next(counter) % len(_MOCK_WORDS)]
            suffix = "\n" if (i + 1) % 12 == 0 else " "
            if self.delay:
                time.sleep(self.delay)
            yield word + suffix


# --------------------------------------------------------------------------- #
# Real Apple Silicon inference via mlx-lm.
# --------------------------------------------------------------------------- #


class MLXEngine(BaseEngine):
    """Runs any HuggingFace causal LM through mlx-lm on unified memory."""

    name = "mlx"

    def __init__(self, model_id: str):
        from mlx_lm import load  # imported here so mock mode needs no MLX

        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)

    def count_tokens(self, messages: list[Message]) -> int:
        return len(self.tokenizer.apply_chat_template(messages))

    def _stream(self, messages, *, max_tokens, temperature):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        sampler = make_sampler(temp=temperature)
        for response in stream_generate(
            self.model, self.tokenizer, prompt=prompt,
            max_tokens=max_tokens, sampler=sampler,
        ):
            yield response.text


def build_engine(engine: str, model_id: str) -> BaseEngine:
    """Factory used by the server and CLI tools."""
    if engine == "mock":
        return MockEngine(model_id=model_id)
    if engine == "mlx":
        return MLXEngine(model_id)
    raise ValueError(f"unknown engine: {engine!r} (expected 'mock' or 'mlx')")
