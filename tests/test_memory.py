from nanoserve.memory import GIB, kv_cache_bytes, report, weights_bytes


def test_weights_bytes():
    # 27B params at 4 bits = 27e9 * 4/8 bytes = 13.5e9
    assert weights_bytes(27, 4) == int(13_500_000_000)
    # 8-bit halves it again vs 16-bit
    assert weights_bytes(1, 8) * 2 == weights_bytes(1, 16)


def test_kv_cache_bytes():
    # 2 (K+V) * layers * kv_heads * head_dim * dtype * seq * batch
    got = kv_cache_bytes(n_layers=64, n_kv_heads=8, head_dim=128,
                         seq_len=1024, batch=1, dtype_bytes=2)
    expected = 2 * 64 * 8 * 128 * 2 * 1024
    assert got == expected


def test_kv_cache_scales_linearly_with_batch_and_context():
    base = kv_cache_bytes(10, 4, 64, 1000)
    assert kv_cache_bytes(10, 4, 64, 2000) == 2 * base
    assert kv_cache_bytes(10, 4, 64, 1000, batch=4) == 4 * base


def test_report_flags_overflow():
    text = report(params_billion=30, bits=16, n_layers=64, n_kv_heads=8,
                  head_dim=128, seq_lens=[1024], budget_gb=48.0)
    assert "NO" in text          # 60GiB of weights cannot fit in 48GiB

    text = report(params_billion=27, bits=4, n_layers=64, n_kv_heads=8,
                  head_dim=128, seq_lens=[32768], budget_gb=48.0)
    total_row = [ln for ln in text.splitlines() if "32768" in ln][0]
    assert "yes" in total_row
