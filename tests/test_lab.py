"""Lab loop tests — spec -> run -> aggregate -> score -> report.

Runs entirely on the mock engine, so these pass on any laptop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanoserve import lab, report

pytest.importorskip("tomli") if lab.tomllib is None else None


SPEC = """\
id = "test-exp"
title = "Test experiment"
question = "does it work"
hypothesis = "yes"

[prediction]
metric = "decode_tps"
x = "prompt_tokens_target"
reasoning = "mock engine is rate-limited, so flat"

[prediction.values]
16 = 40.0

[sweep]
engine = "mock"
prompt_tokens = [16, 32]
max_tokens = [8]
repeats = 2
warmup = 1
label = "t"
"""


@pytest.fixture()
def spec(tmp_path, monkeypatch):
    exp_dir = tmp_path / "experiments"
    exp_dir.mkdir()
    path = exp_dir / "test-exp.toml"
    path.write_text(SPEC)
    monkeypatch.setattr(lab, "EXPERIMENTS_DIR", exp_dir)
    monkeypatch.setattr(lab, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(report, "RESULTS_DIR", tmp_path / "results")
    return lab.Spec.load(path)


def test_spec_parses_prediction(spec):
    assert spec.id == "test-exp"
    assert spec.x_key == "prompt_tokens_target"
    assert spec.y_key == "decode_tps"
    assert spec.predicted() == {16.0: 40.0}


def test_find_spec_by_prefix(spec):
    assert lab.find_spec("test").id == "test-exp"
    with pytest.raises(SystemExit):
        lab.find_spec("nope")


def test_run_writes_rows_and_env(spec):
    runs = lab.run_experiment(spec)
    rows = [json.loads(l) for l in runs.read_text().splitlines() if l.strip()]

    # 2 prompt lengths x (2 repeats + 1 warmup)
    assert len(rows) == 6
    assert sum(r["warmup"] for r in rows) == 2
    assert all(r["experiment_id"] == "test-exp" for r in rows)
    assert [r["order"] for r in rows] == list(range(6))

    env = json.loads((runs.parent / "env.json").read_text())
    assert env["engine"] == "mock"
    assert env["spec"]["id"] == "test-exp"


def test_aggregate_excludes_warmups(spec):
    runs = lab.run_experiment(spec)
    rows = [json.loads(l) for l in runs.read_text().splitlines() if l.strip()]
    agg = report.aggregate(rows, "prompt_tokens_target")

    assert len(agg) == 2                      # one per prompt length
    assert all(a["n"] == 2 for a in agg)      # warmups dropped
    assert all("decode_tps" in a for a in agg)


def test_score_prediction_computes_error(spec):
    agg = [{"prompt_tokens_target": 16, "decode_tps": 50.0, "label": "t"}]
    scored = report.score_prediction(agg, spec)
    assert len(scored) == 1
    assert scored[0]["error_pct"] == 25.0     # 50 measured vs 40 predicted
    assert scored[0]["verdict"] == "over"


def test_report_builds_markdown_with_scorecard(spec):
    lab.run_experiment(spec)
    path = report.report("test-exp")
    text = path.read_text()

    assert "# test-exp — Test experiment" in text
    assert "### Scorecard" in text
    assert "## What I learned" in text
    assert "Conditions" in text


def test_status_log_gets_a_row(spec, tmp_path):
    lab.run_experiment(spec)
    report.report("test-exp")
    log = tmp_path / "experiments" / "STATUS.md"
    assert log.exists()
    assert "test-exp" in log.read_text()


def test_missing_runs_is_a_clear_error(spec):
    with pytest.raises(SystemExit):
        report.report("test-exp")
