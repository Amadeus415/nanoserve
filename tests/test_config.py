from nanoserve.config import (
    BASELINE_MODEL,
    MLX_MODEL,
    Settings,
    default_model_id,
    load_settings,
)


def test_qwen3_14b_4bit_is_the_default_model(monkeypatch):
    monkeypatch.delenv("NANOSERVE_MODEL", raising=False)

    assert BASELINE_MODEL == "Qwen/Qwen3-14B"
    assert MLX_MODEL == "mlx-community/Qwen3-14B-4bit"
    assert Settings().model_id == default_model_id()
    assert load_settings().model_id == default_model_id()


def test_thinking_defaults_off_and_can_be_enabled(monkeypatch):
    monkeypatch.delenv("NANOSERVE_THINKING", raising=False)
    assert load_settings().thinking is False

    monkeypatch.setenv("NANOSERVE_THINKING", "true")
    assert load_settings().thinking is True
