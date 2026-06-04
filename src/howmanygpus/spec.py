import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    name: str
    N: float
    L: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    b_bytes: float = 2.0


@dataclass(frozen=True)
class GPUSpec:
    name: str
    flops_per_s: float
    hbm_bytes_per_s: float
    hbm_bytes: float


@dataclass(frozen=True)
class ParallelismSpec:
    tp: int = 1
    pp: int = 1
    replicas: int = 1

    @property
    def gpus_per_replica(self) -> int:
        return self.tp * self.pp

    @property
    def total_gpus(self) -> int:
        return self.gpus_per_replica * self.replicas


@dataclass(frozen=True)
class EfficiencySpec:
    mfu_prefill: float = 0.40
    mfu_decode: float = 0.10
    mbu: float = 0.70
    tp_efficiency: float = 0.90
    hbm_headroom: float = 0.85


@dataclass(frozen=True)
class WorkloadSpec:
    lambda_rps: float
    mean_prompt: float
    cv_prompt: float = 0.5
    mean_output: float = 200.0
    cv_output: float = 0.5
    sim_duration_s: float = 120.0
    seed: int = 0


@dataclass(frozen=True)
class SLOSpec:
    ttft_p95_ms: float = 1000.0
    tpot_p95_ms: float = 100.0


@dataclass
class ClosedFormResult:
    # Closed-form resource floors.
    g_compute: float
    g_bw: float
    g_memory: float
    g_throughput_need: int
    throughput_bottleneck: str
    flops_required_per_s: float
    hbm_bw_required_per_s: float
    resident_memory_required_bytes: float
    usable_flops_per_gpu: float
    usable_hbm_bw_per_gpu: float
    usable_hbm_bytes_per_gpu: float

    # Static model facts.
    weights_gb: float
    kv_per_token_kb: float

    # Layout-specific diagnostics evaluated at the chosen ParallelismSpec.
    layout_total_gpus: int
    min_gpus_per_replica_for_weights: int
    kv_budget_per_replica_tokens: float
    expected_in_flight_kv_per_replica_tokens: float
    layout_fits_weights: bool
    layout_fits_kv: bool
    prefill_time_s: float
    decode_step_s_at_target: float

    notes: list[str] = field(default_factory=list)


@dataclass
class Request:
    id: int
    arrived_at: float
    prompt_len: int
    output_len: int
    tokens_done: int = 0
    kv_tokens: int = 0  # current KV occupancy (prompt + decoded)
    admitted_at: float = -1.0
    first_token_at: float = -1.0
    completed_at: float = -1.0
    replica_id: int = -1

    @property
    def ttft(self) -> float:
        return self.first_token_at - self.arrived_at

    @property
    def tpot(self) -> float:
        if self.tokens_done <= 1 or self.completed_at < 0:
            return float("nan")
        return (self.completed_at - self.first_token_at) / max(1, self.tokens_done - 1)

    @property
    def e2e(self) -> float:
        return self.completed_at - self.arrived_at


@dataclass
class SimResult:
    completed: list[Request]
    queue_depth: list[tuple[float, int]]
    utilization: float
    sim_time_s: float
    preemptions: int = 0
    dropped: int = 0
    mem_samples: list[tuple[float, float, int]] = field(default_factory=list)
    preempt_times: list[float] = field(default_factory=list)
    replica_busy: list[float] = field(default_factory=list)

    def latencies(self) -> dict[str, dict[str, float]]:
        ttfts = np.array(
            [r.ttft * 1000 for r in self.completed if r.first_token_at >= 0]
        )
        tpots = np.array(
            [r.tpot * 1000 for r in self.completed if not math.isnan(r.tpot)]
        )
        e2es = np.array([r.e2e * 1000 for r in self.completed if r.completed_at > 0])
        out = {}
        for name, arr in (("ttft_ms", ttfts), ("tpot_ms", tpots), ("e2e_ms", e2es)):
            if len(arr) == 0:
                out[name] = {
                    "p50": float("nan"),
                    "p95": float("nan"),
                    "p99": float("nan"),
                    "mean": float("nan"),
                    "n": 0,
                }
            else:
                out[name] = {
                    "p50": float(np.percentile(arr, 50)),
                    "p95": float(np.percentile(arr, 95)),
                    "p99": float(np.percentile(arr, 99)),
                    "mean": float(np.mean(arr)),
                    "n": int(len(arr)),
                }
        return out


@dataclass(frozen=True)
class ModelWorkloadMath:
    prompt_tokens: float
    output_tokens: float
    rounded_prompt_tokens: int
    arrival_rate_rps: float
    decode_batch: int
    model_weight_bytes: float
    kv_cache_bytes_per_token: float
    average_decode_context_tokens: float
    prefill_flops_per_request: float
    decode_flops_per_request: float


@dataclass(frozen=True)
class ThroughputMath:
    compute_floor_gpus: float
    bandwidth_floor_gpus: float
    required_gpus: int
    bottleneck: str
    flops_required_per_s: float
    hbm_bw_required_per_s: float
    usable_flops_per_gpu: float
    usable_hbm_bw_per_gpu: float


@dataclass(frozen=True)
class LayoutMath:
    usable_hbm_bytes_per_gpu: float
    min_gpus_per_replica_for_weights: int
    kv_budget_per_replica_tokens: float
    fits_weights: bool


@dataclass(frozen=True)
class ReplicaTimingMath:
    prefill_time_s: float
    decode_step_time_s: float
    average_response_time_s: float


@dataclass(frozen=True)
class MemoryMath:
    expected_in_flight_kv_tokens_total: float
    expected_in_flight_kv_tokens_per_replica: float
    resident_memory_required_bytes: float
    memory_floor_gpus: float
    fits_kv: bool
