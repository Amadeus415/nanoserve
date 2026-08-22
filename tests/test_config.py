from nanoserve.config import BASELINE_MODEL, Settings, load_settings


def test_qwen3_14b_is_the_default_model(monkeypatch):
    monkeypatch.delenv("NANOSERVE_MODEL", raising=False)

    assert BASELINE_MODEL == "Qwen/Qwen3-14B"
    assert Settings().model_id == BASELINE_MODEL
    assert load_settings().model_id == BASELINE_MODEL
