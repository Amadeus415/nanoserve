"""Central configuration.

Everything is overridable via environment variables so you can run
experiments without editing code:

    NANOSERVE_ENGINE=mock|mlx
    NANOSERVE_MODEL=<huggingface repo id>
    NANOSERVE_THINKING=true|false
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# Official BF16 checkpoint. Quantized MLX builds use the same model family;
# see scripts/download.py --help.
BASELINE_MODEL = "Qwen/Qwen3-14B"
MLX_MODEL = "mlx-community/Qwen3-14B-4bit"
LOCAL_MLX_MODEL = (
    Path(__file__).resolve().parents[1] / "weights" / "mlx-community__Qwen3-14B-4bit"
)
# Tiny checkpoint used for smoke-testing the whole stack on any laptop.
DEV_TINY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def default_model_id() -> str:
    """Prefer downloaded weights, otherwise let MLX fetch the 4-bit model."""
    if (LOCAL_MLX_MODEL / "config.json").is_file():
        return str(LOCAL_MLX_MODEL)
    return MLX_MODEL


@dataclass
class Settings:
    """Runtime settings for the server and benchmark tools."""

    engine: str = "auto"            # auto | mock | mlx
    model_id: str = field(default_factory=default_model_id)
    host: str = "127.0.0.1"
    port: int = 8000

    # default sampling params (requests may override)
    max_tokens: int = 256
    temperature: float = 0.7
    thinking: bool = False

    def resolved_engine(self) -> str:
        """Pick the concrete engine. 'auto' uses MLX if importable."""
        if self.engine != "auto":
            return self.engine
        try:
            import mlx_lm  # noqa: F401
            return "mlx"
        except ImportError:
            return "mock"


def load_settings() -> Settings:
    env = os.environ.get
    return Settings(
        engine=env("NANOSERVE_ENGINE", "auto"),
        model_id=env("NANOSERVE_MODEL", default_model_id()),
        host=env("NANOSERVE_HOST", "127.0.0.1"),
        port=int(env("NANOSERVE_PORT", "8000")),
        max_tokens=int(env("NANOSERVE_MAX_TOKENS", "256")),
        temperature=float(env("NANOSERVE_TEMPERATURE", "0.7")),
        thinking=env("NANOSERVE_THINKING", "false").lower() in {"1", "true", "yes"},
    )
