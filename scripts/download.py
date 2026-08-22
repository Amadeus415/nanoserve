"""Download model weights from HuggingFace into ./weights/<name>.

    python scripts/download.py mlx-community/Qwen3.8-27B-4bit     # target setup
    python scripts/download.py Qwen/Qwen2.5-0.5B-Instruct         # tiny smoke test

Tip: prefer pre-quantized MLX repos (4-bit for a 27B on 48GB). To build your
own from official weights:

    python -m mlx_lm convert --hf-path Qwen/Qwen3.8-27B -q --upload-repo none
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def slugify(repo_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", repo_id)


def main() -> None:
    p = argparse.ArgumentParser(description="download HF model weights")
    p.add_argument("repo", help="e.g. Qwen/Qwen3.8-27B or a mlx-community quant")
    p.add_argument("--dest", default="weights", help="parent directory")
    args = p.parse_args()

    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=args.repo,
        local_dir=str(Path(args.dest) / slugify(args.repo)),
    )
    print(f"done -> {local}")


if __name__ == "__main__":
    main()
