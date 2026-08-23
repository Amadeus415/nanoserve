"""Experiment runner — the lab notebook layer on top of bench.py.

`bench.py` answers "what were the numbers". `lab.py` answers "what did I expect,
what did I get, and under what conditions" — which is the part that teaches you
something.

An experiment is one TOML file in `experiments/`. It records the question, the
hypothesis, and *a numeric prediction made before the run*. Running it captures
the sweep plus machine state; `report.py` scores prediction against measurement.

    python -m nanoserve.lab new 002-output-length --title "Output length vs latency"
    python -m nanoserve.lab list
    python -m nanoserve.lab run 001-context-decode
    python -m nanoserve.lab run 001-context-decode --engine mock   # dry plumbing test

Outputs land in `results/<id>/`:
    runs.jsonl   one row per measured generation (warmups flagged, not dropped)
    env.json     machine, versions, git sha, thermal state, spec snapshot

Nothing here is clever. Read it, change it.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .bench import RunConfig, run_once
from .config import Settings, default_model_id
from .engine import build_engine

try:
    import tomllib                       # py3.11+
except ModuleNotFoundError:  # pragma: no cover - py3.10
    try:
        import tomli as tomllib          # pip install tomli
    except ModuleNotFoundError:
        tomllib = None

GIB = 1024 ** 3
REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
RESULTS_DIR = REPO_ROOT / "results"


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #


@dataclass
class Spec:
    """A parsed experiment TOML file."""

    id: str
    path: Path
    raw: dict
    title: str = ""
    question: str = ""
    hypothesis: str = ""
    prediction: dict = field(default_factory=dict)
    sweep: dict = field(default_factory=dict)
    control: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "Spec":
        if tomllib is None:
            raise SystemExit(
                "lab.py needs a TOML parser: Python 3.11+ (tomllib) or `pip install tomli`"
            )
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return cls(
            id=raw.get("id", path.stem),
            path=path,
            raw=raw,
            title=raw.get("title", ""),
            question=raw.get("question", ""),
            hypothesis=raw.get("hypothesis", ""),
            prediction=raw.get("prediction", {}),
            sweep=raw.get("sweep", {}),
            control=raw.get("control", {}),
        )

    @property
    def x_key(self) -> str:
        """Which swept column is the independent variable on the chart."""
        return self.prediction.get("x", "prompt_tokens_target")

    @property
    def y_key(self) -> str:
        return self.prediction.get("metric", "decode_tps")

    def predicted(self) -> dict[float, float]:
        """{x value -> predicted y}. Empty dict means 'no prediction made'."""
        vals = self.prediction.get("values", {})
        return {float(k): float(v) for k, v in vals.items()}


def find_spec(ident: str) -> Spec:
    """Accept a full id, a filename, or a unique prefix like '001'."""
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    cands = sorted(EXPERIMENTS_DIR.glob("*.toml"))
    exact = [p for p in cands if p.stem == ident or p.name == ident]
    if exact:
        return Spec.load(exact[0])
    pref = [p for p in cands if p.stem.startswith(ident)]
    if len(pref) == 1:
        return Spec.load(pref[0])
    if not pref:
        raise SystemExit(f"no experiment matching {ident!r} in {EXPERIMENTS_DIR}")
    names = ", ".join(p.stem for p in pref)
    raise SystemExit(f"{ident!r} is ambiguous: {names}")


# --------------------------------------------------------------------------- #
# Environment capture — the difference between a measurement and an anecdote
# --------------------------------------------------------------------------- #


def _sh(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return ""


def thermal_state() -> str:
    """Best-effort thermal pressure. A throttled M4 looks like a slow model."""
    if sys.platform != "darwin":
        return "n/a"
    therm = _sh(["pmset", "-g", "therm"])
    for line in therm.splitlines():
        if "CPU_Speed_Limit" in line:
            return line.strip()
    return therm.splitlines()[-1].strip() if therm else "unknown"


def capture_env(spec: Spec, engine_name: str, model_id: str) -> dict:
    versions = {"python": sys.version.split()[0]}
    for mod in ("mlx", "mlx_lm"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            pass
    return {
        "experiment_id": spec.id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": engine_name,
        "model": model_id,
        "machine": {
            "platform": platform.platform(),
            "processor": _sh(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor(),
            "ram_gib": round(int(_sh(["sysctl", "-n", "hw.memsize"]) or 0) / GIB, 1),
            "on_ac_power": "AC Power" in _sh(["pmset", "-g", "ps"]),
        },
        "thermal_before": thermal_state(),
        "versions": versions,
        "git_sha": _sh(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"]),
        "git_dirty": bool(_sh(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])),
        "control_notes": spec.control.get("notes", ""),
        "spec": spec.raw,
    }


# --------------------------------------------------------------------------- #
# Memory instrumentation
# --------------------------------------------------------------------------- #


def _mlx_mem_fns():
    """mlx moved these between namespaces; find whichever exists."""
    try:
        import mlx.core as mx
    except Exception:
        return None, None
    ns = [mx, getattr(mx, "metal", None)]
    get = reset = None
    for n in ns:
        if n is None:
            continue
        get = get or getattr(n, "get_peak_memory", None)
        reset = reset or getattr(n, "reset_peak_memory", None)
    return get, reset


def reset_peak_memory() -> None:
    _, reset = _mlx_mem_fns()
    if reset:
        try:
            reset()
        except Exception:
            pass


def peak_memory_gib() -> float | None:
    """Peak memory for this run, in GiB. None if we can't measure it."""
    get, _ = _mlx_mem_fns()
    if get:
        try:
            return round(get() / GIB, 3)
        except Exception:
            pass
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB
        scale = 1 if sys.platform == "darwin" else 1024
        return round(rss * scale / GIB, 3)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def run_experiment(spec: Spec, *, engine_override: str | None = None,
                   out_dir: Path | None = None) -> Path:
    sw = spec.sweep
    engine_name = engine_override or sw.get("engine", "mock")
    model_id = sw.get("model") or (
        default_model_id() if engine_name == "mlx" else "mock-qwen3-14b"
    )
    settings = Settings(engine=engine_name, model_id=model_id,
                        thinking=bool(sw.get("thinking", False)))

    prompt_tokens = sw.get("prompt_tokens", [128])
    max_tokens = sw.get("max_tokens", [256])
    temperature = float(sw.get("temperature", 0.7))
    repeats = int(sw.get("repeats", 3))
    warmup = int(sw.get("warmup", 1))
    label = sw.get("label", "")

    out_dir = out_dir or (RESULTS_DIR / spec.id)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== {spec.id}: {spec.title}")
    print(f"   engine={engine_name} model={model_id}")
    if spec.predicted():
        print(f"   prediction on {spec.y_key}: {spec.predicted()}")
    else:
        print("   !! no prediction recorded — fill [prediction] before running")

    engine = build_engine(settings.resolved_engine(), settings.model_id,
                          thinking=settings.thinking)
    env = capture_env(spec, engine_name, model_id)

    rows: list[dict] = []
    cells = [(pt, mt) for pt in prompt_tokens for mt in max_tokens]
    total = len(cells) * (repeats + warmup)
    done = 0
    for pt, mt in cells:
        for rep in range(-warmup, repeats):
            cfg = RunConfig(
                engine=engine.name, model=engine.model_id,
                prompt_tokens_target=pt, max_tokens=mt,
                temperature=temperature, repeat=max(rep, 0), label=label,
            )
            reset_peak_memory()
            t0 = time.perf_counter()
            row = run_once(engine, cfg)
            row["wall_s"] = round(time.perf_counter() - t0, 4)
            row["peak_ram_gib"] = peak_memory_gib()
            row["warmup"] = rep < 0            # kept in the file, excluded from stats
            row["experiment_id"] = spec.id
            row["order"] = done                # so you can spot thermal drift
            rows.append(row)
            done += 1
            tag = "warmup " if rep < 0 else ""
            print(f"[{done}/{total}] {tag}ctx~{pt} max_tokens={mt} -> "
                  f"ttft={row['ttft_s']}s decode={row['decode_tps']} tok/s "
                  f"ram={row['peak_ram_gib']}")

    env["thermal_after"] = thermal_state()
    env["n_rows"] = len(rows)

    runs_path = out_dir / "runs.jsonl"
    with open(runs_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with open(out_dir / "env.json", "w") as f:
        json.dump(env, f, indent=2)

    print(f"\nwrote {len(rows)} rows -> {runs_path}")
    print(f"next: python -m nanoserve.report {spec.id}")
    return runs_path


# --------------------------------------------------------------------------- #
# Scaffolding
# --------------------------------------------------------------------------- #

TEMPLATE = '''\
id = "{id}"
title = "{title}"

# ---------------------------------------------------------------------------
# 1. THE QUESTION — falsifiable, with a number in it.
#    Bad:  "measure context vs decode speed"
#    Good: "decode tok/s at 32K context will be 58% of decode tok/s at 128"
# ---------------------------------------------------------------------------
question = """
TODO
"""

hypothesis = """
TODO — the mechanism you think is responsible.
"""

# ---------------------------------------------------------------------------
# 2. THE PREDICTION — fill this in BEFORE you run. This is the whole point.
#    Derive it from arithmetic (see education/LEARNING.md), not from vibes.
# ---------------------------------------------------------------------------
[prediction]
metric = "decode_tps"              # y axis: decode_tps | ttft_s | prefill_tps | peak_ram_gib
x = "prompt_tokens_target"         # x axis: any swept column
reasoning = """
TODO — show the math.
"""

[prediction.values]                # x -> predicted y. Omit any you won't guess.
# 128 = 0.0

# ---------------------------------------------------------------------------
# 3. THE SWEEP — change ONE variable. Two variables = uninterpretable result.
# ---------------------------------------------------------------------------
[sweep]
engine = "mlx"                     # mlx | mock
model = ""                         # blank -> config.default_model_id()
prompt_tokens = [128, 1024, 4096]
max_tokens = [256]
temperature = 0.7
repeats = 3
warmup = 1                         # discarded from stats; cold weights lie
label = ""

[control]
notes = "plugged into AC, lid open, no other heavy apps"
'''


def new_experiment(ident: str, title: str) -> Path:
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    path = EXPERIMENTS_DIR / f"{ident}.toml"
    if path.exists():
        raise SystemExit(f"{path} already exists")
    path.write_text(TEMPLATE.format(id=ident, title=title or ident))
    print(f"created {path}\nedit it — especially [prediction] — then: "
          f"python -m nanoserve.lab run {ident}")
    return path


def list_experiments() -> None:
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    specs = sorted(EXPERIMENTS_DIR.glob("*.toml"))
    if not specs:
        print("no experiments yet — python -m nanoserve.lab new 001-my-experiment")
        return
    for p in specs:
        s = Spec.load(p)
        ran = (RESULTS_DIR / s.id / "runs.jsonl").exists()
        pred = "pred" if s.predicted() else "NO PREDICTION"
        print(f"{'[run]' if ran else '[   ]'} {s.id:<28} {pred:<14} {s.title}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="nanoserve experiment lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_new = sub.add_parser("new", help="scaffold an experiment spec")
    p_new.add_argument("id")
    p_new.add_argument("--title", default="")

    sub.add_parser("list", help="list experiments and whether they've run")

    p_run = sub.add_parser("run", help="run an experiment spec")
    p_run.add_argument("id")
    p_run.add_argument("--engine", default=None,
                       help="override the spec's engine (e.g. mock, to test plumbing)")

    args = p.parse_args(argv)
    if args.cmd == "new":
        new_experiment(args.id, args.title)
    elif args.cmd == "list":
        list_experiments()
    elif args.cmd == "run":
        run_experiment(find_spec(args.id), engine_override=args.engine)


if __name__ == "__main__":
    main()
