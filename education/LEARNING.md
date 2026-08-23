# Learning serving — reading list + experiment protocol

Companion to [`experiments.md`](experiments.md). That file says *what* to measure.
This one says *what to read* and *how to run a run* so it teaches you something.

---

## Reading list

Read in this order. Each one maps to something you can measure on the M4.

### 1. Foundations — read before your next benchmark

**Transformer Inference Arithmetic** — kipply (Carol Chen)
<https://kipp.ly/transformer-inference-arithmetic/>
First-principles latency math: the `2·P` flops rule, KV cache sizing, and
*why* prefill is compute-bound and decode is memory-bandwidth-bound. This is the
single highest-value thing on the list. You should be able to predict your own
decode tok/s from memory bandwidth alone after reading it.

**Transformer Math 101** — EleutherAI
<https://blog.eleuther.ai/transformer-math/>
Same territory, more on memory accounting. Good cross-check on `memory.py`.

### 2. Continuous batching — the core of serving

**Orca: A Distributed Serving System for Transformer-Based Generative Models**
(Yu et al., OSDI '22)
<https://www.usenix.org/conference/osdi22/presentation/yu>
Introduces *iteration-level scheduling* (continuous batching) and *selective
batching*. This is the paper that separates "running a model" from "serving" it.
Read it before you implement concurrency in `server.py`.

### 3. Memory management

**Efficient Memory Management for LLM Serving with PagedAttention**
(Kwon et al., SOSP '23) — the vLLM paper
<https://arxiv.org/abs/2309.06180>
KV cache as OS virtual memory: blocks, fragmentation, copy-on-write sharing.
Explains why naive contiguous KV allocation caps your batch size.

### 4. Scheduling the prefill/decode conflict

**Sarathi-Serve: Taming the Throughput-Latency Tradeoff in LLM Inference**
(Agrawal et al., OSDI '24)
<https://arxiv.org/abs/2403.02310>
(earlier SARATHI paper: <https://arxiv.org/abs/2308.16369>)
Chunked prefill and stall-free scheduling. Read after you've *personally
observed* a long prefill blocking your decode stream — it'll land much harder.

### 5. Production code, after the papers

vLLM's scheduler — `vllm/core/scheduler.py` and `vllm/engine/llm_engine.py`.
Read it with Orca + PagedAttention fresh in your head; you'll recognize the
structures instead of drowning in them.

---

## How to run an experiment so you actually learn

The trap is running sweeps and collecting PNGs. Charts don't teach; *wrong
predictions* teach. Use this loop.

### The loop

**1. Write the question as a falsifiable sentence.**
Not "measure context vs decode speed." Instead:
> "Decode tok/s at 32K context will be X% of decode tok/s at 128 tokens."

**2. Predict the number before you run. Write it down.**
Use the arithmetic from kipply's post. For Qwen3-14B 4-bit:

```
KV bytes/token = 2 × layers × kv_heads × head_dim × dtype_bytes
               = 2 × 40 × 8 × 128 × 2   (fp16 KV)  = 163,840 B ≈ 160 KiB/token
```

So 32K context ≈ 5 GiB of KV cache. Weights ≈ 7 GiB. Every decode step re-reads
weights + KV, so predicted slowdown ≈ (7+5)/7 ≈ 0.58× → **~42% slower**.
Now go find out if you're right.

**3. Change exactly one variable.** One knob per sweep. If you change quant
*and* context in the same run, the result is uninterpretable.

**4. Control the environment.** Same conditions every time, or the noise eats
the signal:

```bash
# plugged into power, lid open, no other heavy apps
sudo pmset -a disablesleep 1
caffeinate -i python -m nanoserve.bench ...
```

- Discard the first run of every sweep (cold weights / page cache).
- `--repeats 3` minimum; report median, not mean.
- Watch thermals — long sweeps throttle the M4 and you'll misread it as a
  context effect. Log `powermetrics` or at least note run order.

**5. Explain the gap between prediction and measurement.**
This is the actual learning step. If you predicted 42% slower and measured 55%,
*something else is happening* — attention compute growth, memory allocator
behavior, throttling. Chase it. Don't log the number and move on.

**6. Complete the status-log row yourself.** `report.py` appends one line to
`experiments/STATUS.md`; replace its `_TODO_` with the surprise. If you can't
write the "what I learned" column without help, you haven't finished the
experiment.

### Suggested order (differs from experiments.md)

| # | Experiment | Why this order |
|---|---|---|
| 1 | Context length → TTFT / decode tok/s | Cheapest, and it validates your mental model of prefill vs decode |
| 2 | Output length → latency breakdown | Cleanly separates the two phases; needs no new code |
| 3 | Quantization sweep (4/8/BF16) | Requires downloads; do it once the harness is trustworthy |
| 4 | **Concurrency 1→16** | The real serving experiment. Needs continuous batching in `server.py` |
| 5 | Model size / architecture comparison | Most download-heavy, least conceptually new |

Concurrency is #4 here rather than late in the roadmap because everything
before it is *model* benchmarking. Scheduling under load is where serving
engineering actually lives.

### Missing instrumentation to add first

`bench.py` doesn't yet capture two things your experiment matrix asks for:

- **Peak RAM** — on unified memory this is the binding constraint. Sample RSS +
  `mlx.core.metal.get_peak_memory()` per run.
- **Quality** — even a crude fixed-prompt scorer, so the quantization sweep has
  a y-axis besides speed.

Add these before the quant sweep or you'll have to re-run it.

### For the concurrency experiment specifically

You need a load generator that holds N streams open, not a sequential loop.
Record per-request TTFT and inter-token latency, then plot:

- offered concurrency → aggregate tok/s (throughput curve, should saturate)
- offered concurrency → p50 / p95 TTFT (latency curve, should hockey-stick)

The point where throughput flattens but p95 keeps climbing is the saturation
knee. Finding that knee on your own laptop is the moment continuous batching
stops being an abstraction.
