import time

from nanoserve.engine import MockEngine


def test_mock_stream_is_deterministic():
    msgs = [{"role": "user", "content": "hi"}]
    a = list(MockEngine(words_per_second=0)._stream(msgs, max_tokens=10, temperature=1.0))
    b = list(MockEngine(words_per_second=0)._stream(msgs, max_tokens=10, temperature=1.0))
    assert a == b
    assert len(a) == 10


def test_mock_generate_stats():
    engine = MockEngine(model_id="mock-test", words_per_second=500)
    msgs = [{"role": "user", "content": "one two three four five"}]
    result = engine.generate(msgs, max_tokens=20, temperature=0.0)

    assert result.stats.prompt_tokens == 5
    assert result.stats.output_tokens == 20
    assert result.stats.ttft_s > 0
    assert result.stats.decode_tps > 0
    d = result.stats.to_dict()
    assert set(d) >= {"engine", "model", "prompt_tokens", "output_tokens",
                      "ttft_s", "total_s", "prefill_tps", "decode_tps"}


def test_mock_delay_slows_decode():
    fast = MockEngine(words_per_second=100_000)
    slow = MockEngine(words_per_second=2_000)
    msgs = [{"role": "user", "content": "hello world"}]

    t0 = time.perf_counter()
    fast.generate(msgs, max_tokens=50, temperature=0.0)
    fast_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    slow.generate(msgs, max_tokens=50, temperature=0.0)
    slow_s = time.perf_counter() - t0

    assert slow_s > fast_s * 3  # artificial latency is reflected in stats
