"""Benchmark harness: measure serving metrics across a parameter grid.

This is the core of your experimentation loop. Each run appends one JSON row
per measurement to an output file, so sweeps compose trivially:

    python -m nanoserve.bench --engine mock --out results/mock.jsonl

    # on the M4, once weights are local:
    python -m nanoserve.bench --engine mlx \
        --model <4-bit model repo or local path> \
        --prompt-tokens 128 1024 4096 16384 \
        --max-tokens 256 --repeats 3 --out results/qwen14b-ctx.jsonl

Metrics captured per run:
    ttft_s       time-to-first-token (dominated by prompt processing)
    prefill_tps  prompt tokens / second
    decode_tps   generated tokens / second
    output_tokens, total_s
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field

from .config import BASELINE_MODEL, Settings
from .engine import BaseEngine, Message, Stats, build_engine


@dataclass
class RunConfig:
    """One cell of the sweep grid."""

    engine: str
    model: str
    prompt_tokens_target: int
    max_tokens: int
    temperature: float
    repeat: int = 0
    label: str = ""                 # free-form tag, e.g. "4bit", "8bit"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v not in ("", {}, None)}


def make_prompt(target_tokens: int) -> list[Message]:
    """Build a chat whose length is ~target_tokens words.

    Word-based approximation is fine for benchmarking; the engine reports the
    token count it actually saw.
    """
    sentence = "The quick brown fox jumps over the lazy dog near the river bank at dawn "
    words_needed = max(target_tokens - 12, 1)
    body = sentence * (words_needed // len(sentence.split()) + 1)
    body = " ".join(body.split()[:words_needed])
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": body + "\n\nSummarize the passage above."},
    ]


def run_once(engine: BaseEngine, cfg: RunConfig) -> dict:
    messages = make_prompt(cfg.prompt_tokens_target)
    result = engine.generate(
        messages, max_tokens=cfg.max_tokens, temperature=cfg.temperature
    )
    row = {**cfg.to_dict(), **result.stats.to_dict()}
    return row


def sweep(engine: BaseEngine, args: argparse.Namespace) -> list[dict]:
    rows: list[dict] = []
    total = len(args.prompt_tokens) * len(args.max_tokens) * args.repeats
    done = 0
    for pt in args.prompt_tokens:
        for mt in args.max_tokens:
            for rep in range(args.repeats):
                cfg = RunConfig(
                    engine=engine.name, model=engine.model_id,
                    prompt_tokens_target=pt, max_tokens=mt,
                    temperature=args.temperature, repeat=rep,
                    label=args.label,
                )
                t0 = time.perf_counter()
                row = run_once(engine, cfg)
                wall = time.perf_counter() - t0
                row["wall_s"] = round(wall, 4)
                rows.append(row)
                done += 1
                print(f"[{done}/{total}] ctx~{pt} tok max_tokens={mt} rep={rep} -> "
                      f"ttft={row['ttft_s']}s decode={row['decode_tps']} tok/s")
    return rows


def write_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows -> {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="nanoserve benchmark runner")
    p.add_argument("--engine", default="mock", choices=["mock", "mlx"])
    p.add_argument("--model", default=None, help="HF repo id or local path (mlx engine)")
    p.add_argument("--prompt-tokens", type=int, nargs="+", default=[16],
                   help="approximate prompt lengths to test")
    p.add_argument("--max-tokens", type=int, nargs="+", default=[64],
                   help="generation lengths to test")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--label", default="", help="tag stored on every row")
    p.add_argument("--out", default="results/bench.jsonl")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = Settings(engine=args.engine, model_id=args.model or (
        BASELINE_MODEL if args.engine == "mlx" else "mock-qwen3-14b"
    ))
    engine = build_engine(settings.resolved_engine(), settings.model_id)
    rows = sweep(engine, args)
    write_jsonl(rows, args.out)


if __name__ == "__main__":
    main()
