"""Turn benchmark JSONL into charts.

    python -m nanoserve.plot results/bench.jsonl            # writes PNGs alongside
    python -m nanoserve.plot results/a.jsonl results/b.jsonl --out results/

Charts produced (the two relationships that matter most when serving):
    1. decode throughput vs context length   -> "does long context slow generation?"
    2. time-to-first-token vs context length -> prefill cost curve
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_rows(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def group_series(rows: list[dict], x_key: str, y_key: str) -> dict[str, tuple[list, list]]:
    """Group rows by max_tokens so each becomes its own line on the chart."""
    series: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        series[f"max_tokens={r['max_tokens']}"].append(r)
    out = {}
    for label, group in series.items():
        group.sort(key=lambda r: r[x_key])
        xs = [r[x_key] for r in group]
        ys = [r[y_key] for r in group]
        out[label] = (xs, ys)
    return out


def _plot(rows, x_key, y_key, title, xlabel, ylabel, out_path, logx=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, (xs, ys) in group_series(rows, x_key, y_key).items():
        ax.plot(xs, ys, marker="o", label=label)
    if logx and xs and min(xs) > 0:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


def make_charts(rows: list[dict], out_dir: str, stem: str = "bench") -> list[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    made = []
    targets = [
        ("prompt_tokens_target", "decode_tps",
         "Decode throughput vs prompt length", "approx. prompt tokens", "tokens/sec (decode)",
         f"{stem}_decode_tps.png"),
        ("prompt_tokens_target", "ttft_s",
         "Time to first token vs prompt length", "approx. prompt tokens", "TTFT (s)",
         f"{stem}_ttft.png"),
    ]
    for x_key, y_key, title, xlabel, ylabel, fname in targets:
        path = str(Path(out_dir) / fname)
        _plot(rows, x_key, y_key, title, xlabel, ylabel, path)
        made.append(path)
    return made


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="plot benchmark JSONL files")
    p.add_argument("inputs", nargs="+", help="one or more .jsonl files from bench.py")
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)
    rows = load_rows(args.inputs)
    if not rows:
        raise SystemExit("no data rows found")
    make_charts(rows, args.out)


if __name__ == "__main__":
    main()
