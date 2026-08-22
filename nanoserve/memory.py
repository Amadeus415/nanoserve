"""Back-of-envelope memory math for local LLM serving.

The two consumers of memory at inference time:

    1. Weights        params * bytes_per_param
    2. KV cache       grows linearly with context length x batch size

Everything else (activations, buffers) is small for single-request decode on
unified-memory Macs. Use this module to sanity-check configs before waiting
for a multi-GB download:

    python -m nanoserve.memory --params 27 --bits 4 --seq-len 32768
    python -m nanoserve.memory --config-url Qwen/Qwen3.8-27B --bits 4

Formulas (transformer decoder, GQA):
    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
                         ^ K and V          ^ grouped-query attention
"""

from __future__ import annotations

import argparse
import json


GIB = 1024 ** 3


def weights_bytes(params_billion: float, bits: int) -> int:
    """Total bytes for model weights at a given precision."""
    return int(params_billion * 1e9 * bits / 8)


def kv_cache_bytes(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch: int = 1,
    dtype_bytes: int = 2,
) -> int:
    """Bytes of KV cache for `batch` sequences of length seq_len."""
    per_token = 2 * n_layers * n_kv_heads * head_dim * dtype_bytes
    return per_token * seq_len * batch


def arch_from_hf(repo_id: str) -> dict:
    """Fetch config.json for a HF repo and pull the fields we need."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, "config.json")
    with open(path) as f:
        cfg = json.load(f)
    hidden = cfg.get("hidden_size")
    heads = cfg.get("num_attention_heads")
    return {
        "n_layers": cfg.get("num_hidden_layers"),
        "n_kv_heads": cfg.get("num_key_value_heads", heads),
        "head_dim": cfg.get("head_dim", hidden // heads if hidden and heads else None),
    }


def report(
    params_billion: float,
    bits: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_lens: list[int],
    budget_gb: float = 48.0,
    kv_dtype_bytes: int = 2,
    overhead_gb: float = 1.0,
) -> str:
    lines = [
        f"weights: {params_billion}B @ {bits}-bit "
        f"= {weights_bytes(params_billion, bits) / GIB:.1f} GiB",
        "",
        f"{'context':>10} {'kv cache':>12} {'total est.':>12} {'fits ' + str(int(budget_gb)) + 'GiB':>10}",
    ]
    w = weights_bytes(params_billion, bits)
    for seq in seq_lens:
        kv = kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq,
                            dtype_bytes=kv_dtype_bytes)
        total = w + kv + overhead_gb * GIB
        fits = "yes" if total < budget_gb * GIB else "NO"
        lines.append(f"{seq:>10} {kv / GIB:>10.1f}GiB {total / GIB:>10.1f}GiB {fits:>>10}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="estimate serving memory footprint")
    p.add_argument("--params", type=float, default=27.0, help="billions of parameters")
    p.add_argument("--bits", type=int, default=4, help="weight precision in bits")
    p.add_argument("--config-url", default=None,
                   help="HF repo id; pulls real layer/head counts from config.json")
    p.add_argument("--layers", type=int, default=64)
    p.add_argument("--kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--seq-lens", type=int, nargs="+",
                   default=[2048, 8192, 32768, 131072])
    p.add_argument("--budget-gb", type=float, default=48.0)
    args = p.parse_args(argv)

    layers, kv_heads, head_dim = args.layers, args.kv_heads, args.head_dim
    if args.config_url:
        try:
            arch = arch_from_hf(args.config_url)
            layers = arch["n_layers"] or layers
            kv_heads = arch["n_kv_heads"] or kv_heads
            head_dim = arch["head_dim"] or head_dim
            print(f"architecture from {args.config_url}: "
                  f"layers={layers} kv_heads={kv_heads} head_dim={head_dim}")
        except Exception as e:  # offline etc: fall back to defaults
            print(f"could not fetch config ({e}); using assumed geometry")
    print(report(args.params, args.bits, layers, kv_heads, head_dim,
                 args.seq_lens, args.budget_gb))


if __name__ == "__main__":
    main()
