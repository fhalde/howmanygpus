# LLM x GPU sizing toolkit

"GPU poor" is not a lifestyle – it's just a capacity planning mistake.

Running LLMs at scale without thinking about throughput, bandwidth, and KV cache is how you end up either (a) burning money, and (b) under the bridge.

This toolkit helps you avoid both.

It answers a simple question: how many GPUs do you actually need to serve an LLM at your target load? Under the hood, it combines closed-form capacity floors, a discrete-event simulator (simpy), packaged as a streamlit app.

**[Live demo](https://huggingface.co/spaces/faizhalde/howmanygpus)** , **[Blog post](https://fhalde.github.io/posts/sizing/)**

## Quick start

```bash
uv sync
uv run streamlit run src/howmanygpus/main.py
```

## References

The formulas and framing draw on:

- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html) — prefill/decode cost model and serving intuition
- [Modal — Roofline model](https://modal.com/gpu-glossary/perf/roofline-model) — compute vs. memory-bandwidth binding
- [Tensor Economics — LLM inference economics from first principles](https://www.tensoreconomics.com/p/llm-inference-economics-from-first) — capacity planning from hardware specs
