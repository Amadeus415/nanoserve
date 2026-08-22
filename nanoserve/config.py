"""Central configuration.

Everything is overridable via environment variables so you can run
experiments without editing code:

    NANOSERVE_ENGINE=mock|mlx
    NANOSERVE_MODEL=<huggingface repo id>
"""

from __future__ import annotations

import os
from dataclasses import dataclass


# Official BF16 checkpoint. Quantized MLX builds use the same model family;
# see scripts/download.py --help.
BASELINE_MODEL = "Qwen/Qwen3-14B"
# Tiny checkpoint used for smoke-testing the whole stack on any laptop.
DEV_TINY_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@dataclass
class Settings:
    """Runtime settings for the server and benchmark tools."""

    engine: str = "auto"            # auto | mock | mlx
    model_id: str = BASELINE_MODEL
    host: str = "127.0.0.1"
    port: int = 8000

    # default sampling params (requests may override)
    max_tokens: int = 256
    temperature: float = 0.7

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
        model_id=env("NANOSERVE_MODEL", BASELINE_MODEL),
        host=env("NANOSERVE_HOST", "127.0.0.1"),
        port=int(env("NANOSERVE_PORT", "8000")),
        max_tokens=int(env("NANOSERVE_MAX_TOKENS", "256")),
        temperature=float(env("NANOSERVE_TEMPERATURE", "0.7")),
    )
