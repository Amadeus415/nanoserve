# Experiments — nanoserve serving lab

Goal: understand LLM *serving* (not just run local models) by physically
measuring weights, quantization, context/KV cache, prefill vs decode,
batching, latency, and throughput on one machine.

Machine: MacBook Pro M4 Pro, 48GB unified memory.
Engine: MLX / MLX-LM first (Apple Silicon native), llama.cpp later for
comparing serving implementations rather than just models.

## Model strategy

One model family at several sizes/quantizations — change one variable at a time.

**Baseline: Qwen3-14B** (14.8B params, 40 layers, 40,960-token native context,
with official BF16 and MLX-community quantized checkpoints). Why 14B:

- ~4B is too easy — everything fits, tradeoffs are invisible.
- ~70B means only fighting memory, no serving insight.
- 14B is the sweet spot where quantization visibly changes the system:

```text
Qwen3-14B BF16   ~30 GB   starts stressing memory
Qwen3-14B Q8     ~15 GB   comfortable
Qwen3-14B Q4     ~7-10 GB lots of room for context/concurrency
```

(Working figures, not exact runtime usage.)

## The experiment matrix

| Experiment       | Model                      | What you learn                     |
| ---------------- | -------------------------- | ---------------------------------- |
| Small baseline   | Qwen ~4–8B                 | Very fast serving, low memory      |
| Main test model  | **Qwen3-14B**              | Sweet spot for experimentation     |
| Large model      | Qwen ~27–32B               | Memory bandwidth / RAM constraints |
| Quantization A   | Qwen3-14B BF16             | High precision baseline            |
| Quantization B   | Qwen3-14B 8-bit            | Memory vs speed                    |
| Quantization C   | Qwen3-14B 4-bit            | Compression vs quality             |
| Context test     | same model, 4K → 32K       | KV-cache effects                   |
| Concurrency test | same model, 1 → N requests | Throughput vs latency              |

48GB comfortably fits 14B at any quant; Q4 of 27–35B class models (~19–25 GB
weights) leaves headroom for KV cache + OS.

## Metrics to record per run

```text
model, parameter count
quantization          BF16 / 8-bit / 6-bit / 4-bit
prompt tokens, output tokens, context size
RAM usage
TTFT                  time to first token
decode speed          tokens/sec
total latency
requests/sec
quality score
```

(nanoserve's `bench.py` JSONL rows carry most of these; RAM + quality need adding.)

## Experiment 1 — Quantization

Hold everything constant except precision:

```text
Qwen3-14B BF16 / Q8 / Q6 / Q4
```

Plot: quantization → RAM, tok/s, TTFT, quality.
This makes quantization click immediately.

```bash
python -m nanoserve.bench --engine mlx --model <bf16-path> \
    --prompt-tokens 128 1024 --max-tokens 256 --repeats 3 --label bf16 --out results/q14b.jsonl
# repeat per quant, same label convention
```

## Experiment 2 — Model size

```text
Qwen small (~4–8B) → 8B-ish → 14B → 27–32B
```

Plot: parameters → tok/s, parameters → RAM, parameters → quality.
You start seeing why serving economics matter.

## Experiment 3 — Context length

Same model, same quant. Prompts at:

```text
1K → 4K → 8K → 16K → 32K
```

Measure RAM, TTFT, prefill speed, decode speed.

Key insight to verify: **long prompts mainly hammer prefill and KV-cache
memory**, not decode arithmetic.

```bash
python -m nanoserve.bench --engine mlx --model <q4-path> \
    --prompt-tokens 1024 4096 16384 32768 --max-tokens 256 --repeats 3 --out results/q14b-context.jsonl
python -m nanoserve.plot results/q14b-context.jsonl --out results
```

Sanity-check feasibility first with the memory calculator:
`python -m nanoserve.memory --params 14.8 --bits 4 --layers 40 ...`

## Experiment 4 — Output length

Constant input, generate 100 → 500 → 1000 → 4000 tokens.
Separates **prefill latency** from **decode throughput** — the fundamental
distinction in LLM serving.

## Experiment 5 — Concurrency

Simulate 1, 2, 4, 8, 16 simultaneous requests.
Measure requests/sec, aggregate tok/s, TTFT, latency/request.

Closest to real serving engineering: batching, scheduling, and
throughput-vs-latency stop being abstract. (Requires concurrency support in
server.py — roadmap item.)

## Then: add one architecturally different model

Add **Gemma 3 27B** or another ~20–30B model and ask:

> Why does Model A use more memory / run faster even though parameter counts
> are similar?

Brings architecture into play: hidden size, layers, GQA config, attention
implementation, KV heads, vocab size. E.g. Qwen3-14B has 40 query heads but
only 8 KV heads (GQA) — once you measure KV-cache usage, that detail becomes
meaningful rather than trivia.

## Plots to generate

- Quantization → RAM
- Quantization → tok/s
- Model size → tok/s
- Context length → TTFT
- Context length → memory
- Concurrency → throughput
- Concurrency → p95 latency

## Status log

| Date | Experiment | Model/quant | Result summary | Artifacts |
|------|-----------|-------------|----------------|-----------|
| | | | | |
