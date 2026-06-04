import math

from howmanygpus.spec import (
    ClosedFormResult,
    EfficiencySpec,
    GPUSpec,
    LayoutMath,
    MemoryMath,
    ModelSpec,
    ModelWorkloadMath,
    ParallelismSpec,
    ReplicaTimingMath,
    ThroughputMath,
    WorkloadSpec,
)

# L: number of layers
# N: number of model parameters
# n_kv_heads: number of KV heads
# d_head: dimension of each head
# b_bytes: bytes per parameter
# n_heads: number of attention heads
# mfu_prefill: prefill MFU
# mfu_decode: decode MFU
# tp_efficiency: TP efficiency
# hbm_headroom: HBM headroom
# hbm_bytes: HBM bytes
# hbm_bytes_per_s: HBM bytes per second
# flops_per_s: FLOPS per second
# flops_per_s_per_gpu: FLOPS per second per GPU

# https://horace.io/brrr_intro.html
# https://modal.com/gpu-glossary/perf/roofline-model
# https://www.tensoreconomics.com/p/llm-inference-economics-from-first


def kv_bytes_per_token(m: ModelSpec) -> float:
    return 2.0 * m.L * m.n_kv_heads * m.d_head * m.b_bytes


def weights_bytes(m: ModelSpec) -> float:
    return m.N * m.b_bytes


def prefill_flops(m: ModelSpec, prompt_len: int) -> float:
    dense = 2.0 * m.N * prompt_len
    attn = 4.0 * m.L * prompt_len * prompt_len * m.n_heads * m.d_head
    return dense + attn


def decode_flops_per_step(m: ModelSpec, batch_size: int) -> float:
    return 2.0 * m.N * batch_size


def replica_flops_per_s(
    g: GPUSpec, p: ParallelismSpec, eff_mfu: float, e: EfficiencySpec
) -> float:
    return g.flops_per_s * p.gpus_per_replica * eff_mfu * e.tp_efficiency


def replica_bw_per_s(g: GPUSpec, p: ParallelismSpec, e: EfficiencySpec) -> float:
    return g.hbm_bytes_per_s * p.gpus_per_replica * e.mbu * e.tp_efficiency


def replica_hbm_bytes(g: GPUSpec, p: ParallelismSpec, e: EfficiencySpec) -> float:
    return g.hbm_bytes * p.gpus_per_replica * e.hbm_headroom


def kv_budget_tokens(
    m: ModelSpec, g: GPUSpec, p: ParallelismSpec, e: EfficiencySpec
) -> float:
    free = replica_hbm_bytes(g, p, e) - weights_bytes(m)
    return max(0.0, free) / kv_bytes_per_token(m)


def model_workload_math(
    m: ModelSpec, w: WorkloadSpec, decode_batch: int
) -> ModelWorkloadMath:
    prompt_tokens = w.mean_prompt
    output_tokens = w.mean_output
    rounded_prompt_tokens = round(prompt_tokens)
    model_weight_bytes = weights_bytes(m)
    kv_cache_bytes_per_token = kv_bytes_per_token(m)
    average_decode_context_tokens = prompt_tokens + output_tokens / 2.0

    return ModelWorkloadMath(
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        rounded_prompt_tokens=rounded_prompt_tokens,
        arrival_rate_rps=w.lambda_rps,
        decode_batch=max(1, decode_batch),
        model_weight_bytes=model_weight_bytes,
        kv_cache_bytes_per_token=kv_cache_bytes_per_token,
        average_decode_context_tokens=average_decode_context_tokens,
        prefill_flops_per_request=prefill_flops(m, rounded_prompt_tokens),
        decode_flops_per_request=decode_flops_per_step(m, 1) * output_tokens,
    )


def decode_bytes_per_output_token(facts: ModelWorkloadMath) -> float:
    """Approximate HBM bytes read per generated token during decode.

    Formula:
        weights / decode_batch + average_context * kv_bytes_per_token

    The batch shares one model-weight read, while every sequence reads its own
    KV cache over its current context. `average_context` is approximated as
    prompt + output / 2 because generation grows from P to P + O.
    """
    weight_read_bytes_per_output_token = facts.model_weight_bytes / facts.decode_batch
    kv_read_bytes_per_output_token = (
        facts.average_decode_context_tokens * facts.kv_cache_bytes_per_token
    )
    return weight_read_bytes_per_output_token + kv_read_bytes_per_output_token


def throughput_math(
    g: GPUSpec, e: EfficiencySpec, facts: ModelWorkloadMath
) -> ThroughputMath:
    flops_required_per_s = facts.arrival_rate_rps * (
        facts.prefill_flops_per_request + facts.decode_flops_per_request
    )

    usable_flops_per_gpu = g.flops_per_s * e.mfu_prefill * e.tp_efficiency
    compute_floor_gpus = (
        flops_required_per_s / usable_flops_per_gpu
        if usable_flops_per_gpu > 0
        else math.inf
    )

    hbm_bw_required_per_s = (
        facts.arrival_rate_rps
        * facts.output_tokens
        * decode_bytes_per_output_token(facts)
    )
    usable_hbm_bw_per_gpu = g.hbm_bytes_per_s * e.mbu * e.tp_efficiency
    bandwidth_floor_gpus = (
        hbm_bw_required_per_s / usable_hbm_bw_per_gpu
        if usable_hbm_bw_per_gpu > 0
        else math.inf
    )

    resource_floors = {
        "compute": compute_floor_gpus,
        "bandwidth": bandwidth_floor_gpus,
    }
    required_gpus = max(1, math.ceil(max(resource_floors.values())))
    bottleneck = max(resource_floors, key=resource_floors.get)

    return ThroughputMath(
        compute_floor_gpus=compute_floor_gpus,
        bandwidth_floor_gpus=bandwidth_floor_gpus,
        required_gpus=required_gpus,
        bottleneck=bottleneck,
        flops_required_per_s=flops_required_per_s,
        hbm_bw_required_per_s=hbm_bw_required_per_s,
        usable_flops_per_gpu=usable_flops_per_gpu,
        usable_hbm_bw_per_gpu=usable_hbm_bw_per_gpu,
    )


def layout_math(
    g: GPUSpec, p: ParallelismSpec, e: EfficiencySpec, facts: ModelWorkloadMath
) -> LayoutMath:
    usable_hbm_bytes_per_gpu = g.hbm_bytes * e.hbm_headroom
    min_gpus_per_replica_for_weights = max(
        1, math.ceil(facts.model_weight_bytes / usable_hbm_bytes_per_gpu)
    )

    usable_hbm_bytes_per_replica = p.gpus_per_replica * usable_hbm_bytes_per_gpu
    fits_weights = usable_hbm_bytes_per_replica >= facts.model_weight_bytes
    kv_budget_per_replica_tokens = (
        max(0.0, usable_hbm_bytes_per_replica - facts.model_weight_bytes)
        / facts.kv_cache_bytes_per_token
    )

    return LayoutMath(
        usable_hbm_bytes_per_gpu=usable_hbm_bytes_per_gpu,
        min_gpus_per_replica_for_weights=min_gpus_per_replica_for_weights,
        kv_budget_per_replica_tokens=kv_budget_per_replica_tokens,
        fits_weights=fits_weights,
    )


def replica_timing_math(
    m: ModelSpec,
    g: GPUSpec,
    p: ParallelismSpec,
    e: EfficiencySpec,
    facts: ModelWorkloadMath,
) -> ReplicaTimingMath:
    decode_bytes_per_step = (
        facts.model_weight_bytes
        + facts.decode_batch
        * facts.average_decode_context_tokens
        * facts.kv_cache_bytes_per_token
    )
    decode_time_bw_s = decode_bytes_per_step / max(1.0, replica_bw_per_s(g, p, e))
    decode_time_compute_s = decode_flops_per_step(m, facts.decode_batch) / max(
        1.0, replica_flops_per_s(g, p, e.mfu_decode, e)
    )
    decode_step_time_s = max(decode_time_bw_s, decode_time_compute_s)

    prefill_time_s = facts.prefill_flops_per_request / max(
        1.0, replica_flops_per_s(g, p, e.mfu_prefill, e)
    )
    average_response_time_s = prefill_time_s + (
        facts.output_tokens * decode_step_time_s
    )

    return ReplicaTimingMath(
        prefill_time_s=prefill_time_s,
        decode_step_time_s=decode_step_time_s,
        average_response_time_s=average_response_time_s,
    )


def memory_math(
    p: ParallelismSpec,
    layout: LayoutMath,
    timing: ReplicaTimingMath,
    facts: ModelWorkloadMath,
) -> MemoryMath:
    expected_in_flight_requests = (
        facts.arrival_rate_rps * timing.average_response_time_s
    )
    expected_in_flight_kv_tokens_total = (
        expected_in_flight_requests * facts.average_decode_context_tokens
    )
    expected_in_flight_kv_tokens_per_replica = expected_in_flight_kv_tokens_total / max(
        1, p.replicas
    )

    resident_memory_required_bytes = (
        p.replicas * facts.model_weight_bytes
        + expected_in_flight_kv_tokens_total * facts.kv_cache_bytes_per_token
    )
    memory_floor_gpus = (
        resident_memory_required_bytes / layout.usable_hbm_bytes_per_gpu
        if layout.usable_hbm_bytes_per_gpu > 0
        else math.inf
    )
    fits_kv = layout.fits_weights and (
        layout.kv_budget_per_replica_tokens >= expected_in_flight_kv_tokens_per_replica
    )

    return MemoryMath(
        expected_in_flight_kv_tokens_total=expected_in_flight_kv_tokens_total,
        expected_in_flight_kv_tokens_per_replica=expected_in_flight_kv_tokens_per_replica,
        resident_memory_required_bytes=resident_memory_required_bytes,
        memory_floor_gpus=memory_floor_gpus,
        fits_kv=fits_kv,
    )


def closed_form_sizing(
    m: ModelSpec,
    g: GPUSpec,
    p: ParallelismSpec,
    e: EfficiencySpec,
    w: WorkloadSpec,
    target_batch: int = 32,
) -> ClosedFormResult:
    notes: list[str] = []
    model_math = model_workload_math(m, w, target_batch)
    throughput = throughput_math(g, e, model_math)
    layout = layout_math(g, p, e, model_math)
    timing = replica_timing_math(m, g, p, e, model_math)
    memory = memory_math(p, layout, timing, model_math)

    if not layout.fits_weights:
        # notes can be structured data
        notes.append(
            f"Layout does not fit model weights: need at least "
            f"{layout.min_gpus_per_replica_for_weights} GPU(s) per replica, "
            f"you have {p.gpus_per_replica}."
        )
    elif not memory.fits_kv:
        notes.append(
            f"Layout fits weights, but KV budget per replica "
            f"({layout.kv_budget_per_replica_tokens:,.0f} tok) is below "
            f"expected in-flight KV "
            f"({memory.expected_in_flight_kv_tokens_per_replica:,.0f} tok). "
            "Add replicas or increase TP/PP."
        )
    if p.total_gpus < throughput.required_gpus:
        notes.append(
            f"Layout has {p.total_gpus} GPUs but the estimate needs "
            f"~{throughput.required_gpus}. Simulator will show queueing."
        )

    return ClosedFormResult(
        g_compute=throughput.compute_floor_gpus,
        g_bw=throughput.bandwidth_floor_gpus,
        g_memory=memory.memory_floor_gpus,
        g_throughput_need=throughput.required_gpus,
        throughput_bottleneck=throughput.bottleneck,
        flops_required_per_s=throughput.flops_required_per_s,
        hbm_bw_required_per_s=throughput.hbm_bw_required_per_s,
        resident_memory_required_bytes=memory.resident_memory_required_bytes,
        usable_flops_per_gpu=throughput.usable_flops_per_gpu,
        usable_hbm_bw_per_gpu=throughput.usable_hbm_bw_per_gpu,
        usable_hbm_bytes_per_gpu=layout.usable_hbm_bytes_per_gpu,
        weights_gb=model_math.model_weight_bytes / 1e9,
        kv_per_token_kb=model_math.kv_cache_bytes_per_token / 1e3,
        layout_total_gpus=p.total_gpus,
        min_gpus_per_replica_for_weights=layout.min_gpus_per_replica_for_weights,
        kv_budget_per_replica_tokens=layout.kv_budget_per_replica_tokens,
        expected_in_flight_kv_per_replica_tokens=(
            memory.expected_in_flight_kv_tokens_per_replica
        ),
        layout_fits_weights=layout.fits_weights,
        layout_fits_kv=memory.fits_kv,
        prefill_time_s=timing.prefill_time_s,
        decode_step_s_at_target=timing.decode_step_time_s,
        notes=notes,
    )
