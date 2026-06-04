from __future__ import annotations

from dataclasses import replace

from howmanygpus.simulation import simulate
from howmanygpus.sizing import closed_form_sizing
from howmanygpus.spec import (
    EfficiencySpec,
    GPUSpec,
    ModelSpec,
    ParallelismSpec,
    WorkloadSpec,
)


def sweep_gpus_vs_lambda(
    m: ModelSpec,
    g: GPUSpec,
    p: ParallelismSpec,
    e: EfficiencySpec,
    w: WorkloadSpec,
    target_batch: int,
    lambdas: list[float],
) -> list[dict]:
    """Closed-form GPU floors as a function of arrival rate."""
    rows = []
    for lm in lambdas:
        cf = closed_form_sizing(
            m, g, p, e, replace(w, lambda_rps=float(lm)), target_batch=target_batch
        )
        rows.append(
            {
                "lambda_rps": round(float(lm), 2),
                "compute": cf.g_compute,
                "bandwidth": cf.g_bw,
                "required": float(cf.g_throughput_need),
            }
        )
    return rows


def sweep_gpus_vs_seqlen(
    m: ModelSpec,
    g: GPUSpec,
    p: ParallelismSpec,
    e: EfficiencySpec,
    w: WorkloadSpec,
    target_batch: int,
    seqlens: list[float],
    vary: str = "output",
) -> list[dict]:
    """Closed-form GPU floors vs mean output (or prompt) length."""
    rows = []
    for s in seqlens:
        ww = (
            replace(w, mean_output=float(s))
            if vary == "output"
            else replace(w, mean_prompt=float(s))
        )
        cf = closed_form_sizing(m, g, p, e, ww, target_batch=target_batch)
        rows.append(
            {
                "seq_len": round(float(s), 1),
                "compute": cf.g_compute,
                "bandwidth": cf.g_bw,
                "required": float(cf.g_throughput_need),
            }
        )
    return rows


def sweep_bottleneck_grid(
    m: ModelSpec,
    g: GPUSpec,
    p: ParallelismSpec,
    e: EfficiencySpec,
    w: WorkloadSpec,
    target_batch: int,
    lambdas: list[float],
    outputs: list[float],
) -> list[dict]:
    """Required GPUs and bottleneck over a (lambda, mean_output) grid."""
    rows = []
    for lm in lambdas:
        for o in outputs:
            cf = closed_form_sizing(
                m,
                g,
                p,
                e,
                replace(w, lambda_rps=float(lm), mean_output=float(o)),
                target_batch=target_batch,
            )
            rows.append(
                {
                    "lambda_rps": round(float(lm), 2),
                    "mean_output": int(o),
                    "required": int(cf.g_throughput_need),
                    "bottleneck": cf.throughput_bottleneck,
                }
            )
    return rows


def sweep_latency_vs_load(
    m: ModelSpec,
    g: GPUSpec,
    p: ParallelismSpec,
    e: EfficiencySpec,
    w: WorkloadSpec,
    max_batch: int,
    lambdas: list[float],
    duration: float,
) -> list[dict]:
    """Simulated p95 latencies and goodput as offered load rises."""
    rows = []
    for lm in lambdas:
        ww = replace(w, lambda_rps=float(lm), sim_duration_s=float(duration))
        s = simulate(m, g, p, e, ww, max_batch=max_batch)
        lat = s.latencies()
        completed = lat["ttft_ms"]["n"]
        rows.append(
            {
                "lambda_rps": round(float(lm), 2),
                "ttft_p95_ms": lat["ttft_ms"]["p95"],
                "tpot_p95_ms": lat["tpot_ms"]["p95"],
                "e2e_p95_ms": lat["e2e_ms"]["p95"],
                "offered_rps": round(float(lm), 2),
                "goodput_rps": completed / max(1e-9, duration),
                "preemptions": s.preemptions,
            }
        )
    return rows
