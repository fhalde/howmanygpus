# LLM × GPU sizing toolkit

How many GPUs do you need to serve an LLM at a target load? This toolkit answers that from first principles — closed-form capacity floors, a discrete-event simulator, and parameter sweeps — in a single Streamlit app.

**[Live demo](https://howmanygpus.streamlit.app/)** · **[Blog post](https://fhalde.github.io/posts/sizing/)**

## What it does

Pick a model, GPU, parallelism layout, and workload. The app reports whether your configuration is compute-bound, bandwidth-bound, or memory-limited, and how many GPUs you need to sustain the offered load.

| View | What you get |
|------|--------------|
| **Estimate** | Instant throughput floors (compute vs. HBM bandwidth) plus residency checks for weights and KV cache. Assumes steady average load with no queueing. |
| **Simulation** | Discrete-event simulator (SimPy) with Poisson arrivals, prefill/decode scheduling, KV-pressure preemption, and TTFT / TPOT / E2E latency CDFs against SLOs. |
| **Sweeps** | Required GPUs vs. arrival rate and output length; bottleneck heatmap; optional latency-vs-load sweep to find the knee in goodput. |
| **Glossary** | Full formula reference for every metric in the Estimate tab. |

## Quick start

Requires Python ≥ 3.12.

```bash
uv sync
uv run streamlit run src/howmanygpus/main.py
```

Or with pip:

```bash
pip install -e .
streamlit run src/howmanygpus/main.py
```

## Presets

**Models:** Llama-3 (8B / 70B / 405B), Mistral-7B, Qwen2.5-72B — or enter custom architecture fields.

**GPUs:** A100-80GB, H100-80GB, H200-141GB, B200, MI300X — or enter custom peak FLOPs, HBM bandwidth, and capacity.

All presets support dtype selection (BF16/FP16, FP8, INT4 weights).

## References

The formulas and framing draw on:

- [Horace He — *We're Here For A Good Time, Not A Long Time* (BRRR)](https://horace.io/brrr_intro.html) — prefill/decode cost model and serving intuition
- [Modal — Roofline model](https://modal.com/gpu-glossary/perf/roofline-model) — compute vs. memory-bandwidth binding
- [Tensor Economics — LLM inference economics from first principles](https://www.tensoreconomics.com/p/llm-inference-economics-from-first) — capacity planning from hardware specs
