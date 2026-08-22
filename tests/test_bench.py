from nanoserve.bench import RunConfig, make_prompt, run_once
from nanoserve.engine import MockEngine


def test_make_prompt_hits_target_length():
    msgs = make_prompt(100)
    words = len(msgs[-1]["content"].split())
    assert abs(words - 100) <= 12
    assert msgs[0]["role"] == "system"


def test_run_once_row_shape():
    engine = MockEngine(model_id="mock-qwen3-14b", words_per_second=50_000)
    cfg = RunConfig(engine="mock", model="mock-qwen3-14b", prompt_tokens_target=20,
                    max_tokens=16, temperature=0.0, label="unit")
    row = run_once(engine, cfg)
    for key in ("engine", "model", "prompt_tokens_target", "max_tokens",
                "ttft_s", "decode_tps", "prefill_tps", "output_tokens"):
        assert key in row, f"missing {key}"
    assert row["label"] == "unit"
    assert row["output_tokens"] <= 16
