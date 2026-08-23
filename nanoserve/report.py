"""Score an experiment: prediction vs measurement, charts, markdown report.

    python -m nanoserve.report 001-context-decode

Reads `results/<id>/runs.jsonl` + `env.json` (written by lab.py) and produces:

    results/<id>/report.md      the writeup, including a prediction scorecard
    results/<id>/*.png          measured-vs-predicted, TTFT, RAM, drift check

The prediction scorecard is the reason this file exists. A chart of what
happened is a record; the gap between what you expected and what happened is
the thing that teaches you the system.
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

from .lab import RESULTS_DIR, Spec, find_spec

METRICS = ["decode_tps", "ttft_s", "prefill_tps", "total_s", "peak_ram_gib"]


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def load_runs(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def aggregate(rows: list[dict], x_key: str) -> list[dict]:
    """Median over repeats, warmups excluded. Median not mean: one thermal
    stall shouldn't move the number."""
    live = [r for r in rows if not r.get("warmup")]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in live:
        groups[(r.get("label", ""), r[x_key], r.get("max_tokens"))].append(r)

    out = []
    for (label, x, mt), group in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        agg = {"label": label, x_key: x, "max_tokens": mt, "n": len(group)}
        for m in METRICS:
            vals = [r[m] for r in group if r.get(m) is not None]
            if not vals:
                continue
            agg[m] = round(stats.median(vals), 4)
            agg[f"{m}_min"] = round(min(vals), 4)
            agg[f"{m}_max"] = round(max(vals), 4)
            agg[f"{m}_spread_pct"] = (
                round(100 * (max(vals) - min(vals)) / stats.median(vals), 1)
                if stats.median(vals) else 0.0
            )
        # what the tokenizer actually saw, vs what we asked for
        pt = [r["prompt_tokens"] for r in group if r.get("prompt_tokens")]
        if pt:
            agg["prompt_tokens_actual"] = int(stats.median(pt))
        out.append(agg)
    return out


def score_prediction(agg: list[dict], spec: Spec) -> list[dict]:
    """Compare each predicted point to its measured median."""
    pred = spec.predicted()
    if not pred:
        return []
    x_key, y_key = spec.x_key, spec.y_key
    scored = []
    for row in agg:
        x = float(row[x_key])
        if x not in pred:
            continue
        p, actual = pred[x], row.get(y_key)
        if actual is None:
            continue
        err = (actual - p) / p * 100 if p else float("nan")
        scored.append({
            "x": x, "predicted": p, "measured": actual,
            "error_pct": round(err, 1),
            "verdict": "close" if abs(err) < 10 else ("over" if err > 0 else "under"),
        })
    return scored


# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #


def _ax_setup(ax, title, xlabel, ylabel, logx):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()


def chart_metric(agg, spec, metric, out_path, ylabel, with_prediction=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_key = spec.x_key
    have = [r for r in agg if r.get(metric) is not None]
    if not have:
        return None

    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in have:
        by_label[r.get("label") or "run"].append(r)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for label, group in by_label.items():
        group.sort(key=lambda r: r[x_key])
        xs = [r[x_key] for r in group]
        ys = [r[metric] for r in group]
        lo = [r[metric] - r.get(f"{metric}_min", r[metric]) for r in group]
        hi = [r.get(f"{metric}_max", r[metric]) - r[metric] for r in group]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=f"measured ({label})")

    if with_prediction and spec.predicted():
        pred = sorted(spec.predicted().items())
        ax.plot([p[0] for p in pred], [p[1] for p in pred],
                marker="x", linestyle="--", color="crimson", label="predicted")

    logx = min(r[x_key] for r in have) > 0 and len({r[x_key] for r in have}) > 2
    _ax_setup(ax, f"{spec.id}: {ylabel} vs {x_key}", x_key, ylabel, logx)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def chart_drift(rows, out_path):
    """decode_tps against run order. A downward slope here means the machine
    got hot, not that your independent variable did anything."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    live = [r for r in rows if r.get("order") is not None]
    if len(live) < 4:
        return None
    live.sort(key=lambda r: r["order"])
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    warm = [r for r in live if r.get("warmup")]
    real = [r for r in live if not r.get("warmup")]
    ax.plot([r["order"] for r in real], [r["decode_tps"] for r in real],
            marker=".", linestyle="-", label="measured")
    if warm:
        ax.scatter([r["order"] for r in warm], [r["decode_tps"] for r in warm],
                   marker="x", color="gray", label="warmup (excluded)")
    _ax_setup(ax, "Thermal drift check: decode tok/s by run order",
              "run order", "decode tok/s", logx=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #


def _table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(out)


def build_report(spec: Spec, rows: list[dict], agg: list[dict], env: dict,
                 charts: list[str], out_dir: Path) -> Path:
    x_key, y_key = spec.x_key, spec.y_key
    scored = score_prediction(agg, spec)
    m = env.get("machine", {})

    parts = [
        f"# {spec.id} — {spec.title}",
        "",
        f"*Run {env.get('started_at', '?')} · {env.get('engine')} · "
        f"`{env.get('model')}` · git `{env.get('git_sha','?')}"
        f"{'+dirty' if env.get('git_dirty') else ''}`*",
        "",
        "## Question",
        spec.question.strip() or "_not recorded_",
        "",
        "## Hypothesis",
        spec.hypothesis.strip() or "_not recorded_",
        "",
        "## Prediction (made before the run)",
        spec.prediction.get("reasoning", "").strip() or "_no reasoning recorded_",
        "",
    ]

    if scored:
        parts += [
            "### Scorecard",
            "",
            _table([x_key, f"predicted {y_key}", f"measured {y_key}", "error", ""],
                   [[f"{s['x']:g}", f"{s['predicted']:g}", f"{s['measured']:g}",
                     f"{s['error_pct']:+.1f}%", s["verdict"]] for s in scored]),
            "",
        ]
        worst = max(scored, key=lambda s: abs(s["error_pct"]))
        if abs(worst["error_pct"]) >= 10:
            parts += [
                f"> Biggest miss: at {x_key}={worst['x']:g} you predicted "
                f"{worst['predicted']:g} and measured {worst['measured']:g} "
                f"({worst['error_pct']:+.1f}%). **That gap is the experiment.** "
                "Explain it before moving on.",
                "",
            ]
        else:
            parts += ["> Every point within 10%. Your model of the system is "
                      "working — now make a harder prediction.", ""]
    else:
        parts += ["_No prediction was recorded, so this run can only confirm "
                  "what you already saw. Fill in `[prediction.values]` next time._", ""]

    metric_cols = [c for c in METRICS if any(c in r for r in agg)]
    parts += [
        "## Results (median of repeats, warmups excluded)",
        "",
        _table([x_key, "n", "prompt_tok (actual)"] + metric_cols + ["spread"],
               [[r[x_key], r["n"], r.get("prompt_tokens_actual")]
                + [r.get(c) for c in metric_cols]
                + [f"{r.get(y_key + '_spread_pct', 0)}%"]
                for r in agg]),
        "",
    ]

    if charts:
        parts += ["## Charts", ""]
        parts += [f"![{Path(c).stem}]({Path(c).name})" for c in charts]
        parts += [""]

    parts += [
        "## Conditions",
        "",
        _table(["field", "value"], [
            ["machine", m.get("processor", "?")],
            ["RAM", f"{m.get('ram_gib','?')} GiB"],
            ["on AC power", m.get("on_ac_power")],
            ["thermal before", env.get("thermal_before")],
            ["thermal after", env.get("thermal_after")],
            ["versions", ", ".join(f"{k} {v}" for k, v in env.get("versions", {}).items())],
            ["control notes", env.get("control_notes", "")],
        ]),
        "",
        "## What I learned",
        "",
        "<!-- Write this yourself. If you can't, the experiment isn't finished. -->",
        "",
        "- Mechanism behind the gap:",
        "- What this predicts about the next experiment:",
        "- What surprised me:",
        "",
    ]

    path = out_dir / "report.md"
    path.write_text("\n".join(parts))
    return path


def append_status(spec: Spec, scored: list[dict], out_dir: Path) -> None:
    """One line per experiment in experiments/STATUS.md."""
    log = out_dir.parent.parent / "experiments" / "STATUS.md"
    log.parent.mkdir(exist_ok=True)
    if not log.exists():
        log.write_text(
            "# Status log\n\nOne row per run. The **What I learned** column is "
            "yours to write — leave it blank and the run didn't count.\n\n"
            "| Date | Experiment | Prediction hit? | What I learned | Report |\n"
            "|---|---|---|---|---|\n"
        )
    import time as _t
    if scored:
        worst = max(abs(s["error_pct"]) for s in scored)
        verdict = f"within {worst:.0f}%"
    else:
        verdict = "no prediction"
    rel = f"../results/{spec.id}/report.md"
    row = (f"| {_t.strftime('%Y-%m-%d')} | {spec.id} | {verdict} | "
           f"_TODO_ | [report]({rel}) |\n")
    with open(log, "a") as f:
        f.write(row)
    print(f"appended row -> {log}")


# --------------------------------------------------------------------------- #


def report(ident: str) -> Path:
    spec = find_spec(ident)
    out_dir = RESULTS_DIR / spec.id
    runs_path = out_dir / "runs.jsonl"
    if not runs_path.exists():
        raise SystemExit(f"no runs for {spec.id} — python -m nanoserve.lab run {spec.id}")

    rows = load_runs(runs_path)
    env_path = out_dir / "env.json"
    env = json.loads(env_path.read_text()) if env_path.exists() else {}
    agg = aggregate(rows, spec.x_key)

    charts = []
    try:
        primary = chart_metric(agg, spec, spec.y_key, out_dir / f"{spec.id}-{spec.y_key}.png",
                               spec.y_key, with_prediction=True)
        charts.append(primary)
        for metric, ylabel in [("ttft_s", "TTFT (s)"), ("peak_ram_gib", "peak RAM (GiB)")]:
            if metric == spec.y_key:
                continue
            c = chart_metric(agg, spec, metric, out_dir / f"{spec.id}-{metric}.png", ylabel)
            charts.append(c)
        charts.append(chart_drift(rows, out_dir / f"{spec.id}-drift.png"))
    except ImportError:
        print("matplotlib not installed — skipping charts (pip install -e '.[viz]')")
    charts = [c for c in charts if c]

    path = build_report(spec, rows, agg, env, charts, out_dir)
    append_status(spec, score_prediction(agg, spec), out_dir)
    print(f"wrote {path}")
    for c in charts:
        print(f"wrote {c}")
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="score a nanoserve experiment")
    p.add_argument("id", help="experiment id or prefix")
    args = p.parse_args(argv)
    report(args.id)


if __name__ == "__main__":
    main()
