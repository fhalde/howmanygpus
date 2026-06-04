from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from howmanygpus.presets import GPU_PRESETS, MODEL_PRESETS
from howmanygpus.simulation import simulate
from howmanygpus.sizing import closed_form_sizing
from howmanygpus.spec import (
    EfficiencySpec,
    GPUSpec,
    ModelSpec,
    ParallelismSpec,
    SLOSpec,
    WorkloadSpec,
)
from howmanygpus.sweeps import (
    sweep_bottleneck_grid,
    sweep_gpus_vs_lambda,
    sweep_gpus_vs_seqlen,
    sweep_latency_vs_load,
)

# Count axis titles inside the height budget and reserve bottom room so
# Streamlit's fixed-height container never clips the x-axis title.
FIT_AUTOSIZE = {"type": "fit", "contains": "padding"}
CHART_PADDING = {"top": 5, "left": 5, "right": 5, "bottom": 44}


def run_ui() -> None:
    import altair as alt
    import pandas as pd
    import streamlit as st

    def compact_number(value: float, decimals: int = 1) -> str:
        if not math.isfinite(value):
            return "n/a"
        magnitude = abs(value)
        for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
            if magnitude >= scale:
                return f"{value / scale:.{decimals}f}{suffix}"
        if decimals == 0:
            return f"{value:.0f}"
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".")

    st.set_page_config(page_title="LLM x GPU Sizing", layout="wide")
    st.markdown("## LLM x GPU sizing toolkit")

    with st.sidebar:
        st.header("Model")
        m_choice = st.selectbox(
            "Preset", list(MODEL_PRESETS.keys()) + ["Custom"], index=1
        )
        if m_choice == "Custom":
            N = st.number_input("Params (B)", value=70.0, min_value=0.1) * 1e9
            L = st.number_input("Layers", value=80, min_value=1)
            d_model = st.number_input("d_model", value=8192, min_value=64)
            n_heads = st.number_input("n_heads", value=64, min_value=1)
            n_kv_heads = st.number_input("n_kv_heads", value=8, min_value=1)
            d_head = st.number_input("d_head", value=128, min_value=8)
            b_bytes = st.selectbox(
                "dtype bytes",
                [2.0, 1.0, 0.5],
                format_func=lambda b: {
                    2.0: "BF16/FP16",
                    1.0: "FP8",
                    0.5: "INT4 weights",
                }[b],
            )
            m = ModelSpec(
                "Custom",
                N=N,
                L=L,
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                d_head=d_head,
                b_bytes=b_bytes,
            )
        else:
            m = MODEL_PRESETS[m_choice]
            b_bytes = st.selectbox(
                "dtype bytes",
                [2.0, 1.0, 0.5],
                format_func=lambda b: {
                    2.0: "BF16/FP16",
                    1.0: "FP8",
                    0.5: "INT4 weights",
                }[b],
            )
            m = replace(m, b_bytes=b_bytes)

        st.header("GPU")
        g_choice = st.selectbox(
            "Preset", list(GPU_PRESETS.keys()) + ["Custom"], index=1
        )
        if g_choice == "Custom":
            flops_t = st.number_input("TFLOPs (dtype)", value=989.0)
            bw_tb = st.number_input("HBM BW (TB/s)", value=3.35)
            mem_gb = st.number_input("HBM (GB)", value=80.0)
            g = GPUSpec(
                "Custom",
                flops_per_s=flops_t * 1e12,
                hbm_bytes_per_s=bw_tb * 1e12,
                hbm_bytes=mem_gb * 1e9,
            )
        else:
            g = GPU_PRESETS[g_choice]

        st.header("Parallelism")
        tp = st.number_input("TP (tensor parallel)", value=4, min_value=1)
        pp = st.number_input("PP (pipeline parallel)", value=1, min_value=1)
        R = st.number_input("Replicas", value=3, min_value=1)
        p = ParallelismSpec(tp=int(tp), pp=int(pp), replicas=int(R))

        st.header("Efficiency")
        mfu_pre = st.slider("MFU (prefill)", 0.05, 0.8, 0.40, 0.05)
        mfu_dec = st.slider("MFU (decode)", 0.01, 0.5, 0.10, 0.01)
        mbu = st.slider("MBU (decode)", 0.1, 0.95, 0.70, 0.05)
        tp_eff = st.slider("TP efficiency", 0.5, 1.0, 0.90, 0.05)
        headroom = st.slider("HBM headroom", 0.5, 0.95, 0.85, 0.05)
        e = EfficiencySpec(
            mfu_prefill=mfu_pre,
            mfu_decode=mfu_dec,
            mbu=mbu,
            tp_efficiency=tp_eff,
            hbm_headroom=headroom,
        )

        st.header("Workload")
        lam = st.number_input("lambda (req/s)", value=10, min_value=0, step=1)
        mean_p = st.number_input("mean prompt tokens", value=1000)
        cv_p = st.number_input(
            "prompt length spread (CV)",
            value=0.5,
            min_value=0.0,
            help=(
                "How much prompt lengths vary, as coefficient of variation "
                "(std dev / mean). 0 = every prompt identical, 0.5 = moderate "
                "spread, 1.0+ = highly variable. Lengths are drawn from a "
                "lognormal with this spread."
            ),
        )
        mean_o = st.number_input("mean output tokens", value=500)
        cv_o = st.number_input(
            "output length spread (CV)",
            value=0.5,
            min_value=0.0,
            help=(
                "How much output lengths vary, as coefficient of variation "
                "(std dev / mean). 0 = every response identical, 0.5 = moderate "
                "spread, 1.0+ = highly variable. Lengths are drawn from a "
                "lognormal with this spread."
            ),
        )
        target_batch = st.number_input(
            "assumed decode batch",
            value=32,
            min_value=1,
            help=(
                "Average number of requests decoding together, used by the "
                "Estimate tab. Weight reads are amortized across this batch "
                "(weights / batch per token) and it sizes expected in-flight "
                "KV. Higher batch -> better bandwidth efficiency."
            ),
        )
        sim_dur = st.number_input("sim duration (s)", value=60.0, min_value=1.0)
        max_batch = st.number_input("max batch (sim)", value=256, min_value=1)
        seed = st.number_input("seed", value=0)
        w = WorkloadSpec(
            lambda_rps=float(lam),
            mean_prompt=float(mean_p),
            cv_prompt=float(cv_p),
            mean_output=float(mean_o),
            cv_output=float(cv_o),
            sim_duration_s=float(sim_dur),
            seed=int(seed),
        )

        st.header("SLO")
        ttft_slo = st.number_input("TTFT p95 SLO (ms)", value=1000.0, min_value=0.0)
        tpot_slo = st.number_input("TPOT p95 SLO (ms)", value=100.0, min_value=0.0)
        slo = SLOSpec(ttft_p95_ms=float(ttft_slo), tpot_p95_ms=float(tpot_slo))

    cf = closed_form_sizing(m, g, p, e, w, target_batch=int(target_batch))

    def cdf_chart(values, x_title: str, slo_ms: float | None = None):
        arr = np.sort(np.asarray([v for v in values if v == v], dtype=float))
        if arr.size == 0:
            return None
        cdf = np.arange(1, arr.size + 1) / arr.size
        df = pd.DataFrame({"value": arr, "cdf": cdf})
        line = (
            alt.Chart(df)
            .mark_line(color="#4c78a8")
            .encode(
                x=alt.X("value:Q", title=x_title),
                y=alt.Y(
                    "cdf:Q",
                    title="cumulative fraction",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                tooltip=[
                    alt.Tooltip("value:Q", format=".1f", title=x_title),
                    alt.Tooltip("cdf:Q", format=".2f"),
                ],
            )
        )
        layers = [line]
        if slo_ms and slo_ms > 0:
            rule = (
                alt.Chart(pd.DataFrame({"slo": [slo_ms]}))
                .mark_rule(color="#e45756", strokeDash=[5, 5])
                .encode(x="slo:Q")
            )
            layers.append(rule)
        return alt.layer(*layers).properties(
            height=260, autosize=FIT_AUTOSIZE, padding=CHART_PADDING
        )

    def multiline_gpu_chart(rows, x_field, x_title, marker_x=None, hline=None):
        df = pd.DataFrame(rows).melt(
            id_vars=[x_field],
            value_vars=["compute", "bandwidth", "required"],
            var_name="series",
            value_name="gpus",
        )
        line = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(
                x=alt.X(f"{x_field}:Q", title=x_title),
                y=alt.Y("gpus:Q", title="GPUs"),
                color=alt.Color("series:N", title=None),
                tooltip=[x_field, "series", alt.Tooltip("gpus:Q", format=".1f")],
            )
        )
        layers = [line]
        if marker_x is not None:
            layers.append(
                alt.Chart(pd.DataFrame({"x": [marker_x]}))
                .mark_rule(color="#888", strokeDash=[4, 4])
                .encode(x="x:Q")
            )
        if hline is not None:
            layers.append(
                alt.Chart(pd.DataFrame({"y": [hline]}))
                .mark_rule(color="#54a24b", strokeDash=[2, 2])
                .encode(y="y:Q")
            )
        return alt.layer(*layers).properties(
            height=320, autosize=FIT_AUTOSIZE, padding=CHART_PADDING
        )

    view = st.segmented_control(
        "View",
        ["Estimate", "Simulation", "Sweeps", "Glossary"],
        default="Estimate",
        label_visibility="collapsed",
        key="view_mode",
    )

    if view == "Estimate":
        st.caption("Instant throughput floors from formulas – assumes zero queueing.")
        st.markdown("##### Throughput floor")
        col1, col2, col3 = st.columns(3)
        col1.metric("Compute floor (GPUs)", compact_number(cf.g_compute))
        col2.metric("HBM bandwidth floor (GPUs)", compact_number(cf.g_bw))
        col3.metric(
            "Required GPUs",
            compact_number(cf.g_throughput_need, decimals=0),
            delta=f"{cf.throughput_bottleneck}-bound",
        )
        st.info(
            f"Sustaining {cf.flops_required_per_s / 1e15:.2f} PFLOP/s and "
            f"{cf.hbm_bw_required_per_s / 1e12:.2f} TB/s of HBM bandwidth with no "
            f"queueing needs ceil(max({cf.g_compute:.1f}, {cf.g_bw:.1f})) = "
            f"{cf.g_throughput_need} GPUs – the larger of the compute and "
            "bandwidth floors. Memory residency is checked separately below."
        )

        # Build a status line that explains *why* the layout fails, and suggest
        # the smallest fix.
        if not cf.layout_fits_weights:
            status = (
                f":red[NOT OK – weights need GPUs/replica >= "
                f"{cf.min_gpus_per_replica_for_weights} "
                f"(you have {p.gpus_per_replica})]"
            )
        elif not cf.layout_fits_kv:
            status = (
                f":orange[NOT OK – KV tight ({cf.kv_budget_per_replica_tokens:,.0f}"
                f" budget < {cf.expected_in_flight_kv_per_replica_tokens:,.0f} "
                f"expected)]"
            )
        elif p.total_gpus < cf.g_throughput_need:
            need_r = max(1, math.ceil(cf.g_throughput_need / p.gpus_per_replica))
            status = (
                f":orange[NOT OK – under-provisioned "
                f"({p.total_gpus} GPUs < need {cf.g_throughput_need}) "
                f"-> try R >= {need_r}]"
            )
        else:
            status = ":green[OK]"

        st.markdown(f"##### Residency / topology: {status}")
        st.caption(
            "Counts one weight copy per replica plus all in-flight KV. The "
            "per-replica checks below decide whether the topology actually fits."
        )

        kv_budget_str = (
            "does not fit"
            if cf.kv_budget_per_replica_tokens <= 0
            else f"{cf.kv_budget_per_replica_tokens:,.0f} tok"
        )
        layout_rows = [
            ("Weights", f"{cf.weights_gb:.1f} GB"),
            (
                "Min GPUs/replica to hold weights",
                str(cf.min_gpus_per_replica_for_weights),
            ),
            ("KV budget / replica", kv_budget_str),
            (
                "In-flight KV / replica (est.)",
                f"{cf.expected_in_flight_kv_per_replica_tokens:,.0f} tok",
            ),
            ("Usable FLOPs / GPU", f"{cf.usable_flops_per_gpu / 1e12:,.0f} TFLOP/s"),
            ("Usable HBM BW / GPU", f"{cf.usable_hbm_bw_per_gpu / 1e12:.2f} TB/s"),
            ("Usable HBM / GPU", f"{cf.usable_hbm_bytes_per_gpu / 1e9:.1f} GB"),
            (
                "Resident memory needed",
                f"{cf.resident_memory_required_bytes / 1e9:.1f} GB",
            ),
            ("Memory floor (weights + KV)", f"{cf.g_memory:.1f} GPUs"),
            (
                f"Prefill time @ {int(mean_p):,}-tok prompt",
                f"{cf.prefill_time_s * 1000:.0f} ms",
            ),
            (
                f"Decode step time @ batch {int(target_batch)}",
                f"{cf.decode_step_s_at_target * 1000:.1f} ms",
            ),
        ]
        lcol, rcol = st.columns(2)
        for i, (k, v) in enumerate(layout_rows):
            target = lcol if i % 2 == 0 else rcol
            target.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:2px 0;border-bottom:1px solid #eee;font-size:0.85rem;'>"
                f"<span style='opacity:0.7'>{k}</span>"
                f"<span style='font-variant-numeric:tabular-nums'>{v}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        for n in cf.notes:
            st.warning(n)

    elif view == "Simulation":
        st.markdown("##### Simulator")
        with st.spinner(f"Simulating {sim_dur:.0f}s of traffic..."):
            sim = simulate(m, g, p, e, w, max_batch=int(max_batch))
        lat = sim.latencies()
        n = lat["ttft_ms"]["n"]

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Completed", compact_number(n, decimals=0))
        c2.metric("Utilization", f"{compact_number(sim.utilization * 100, 0)}%")
        c3.metric(
            "Arrivals offered",
            compact_number(w.lambda_rps * w.sim_duration_s, decimals=0),
        )
        c4.metric(
            "Throughput (req/s)",
            compact_number(n / max(1e-9, w.sim_duration_s), decimals=2),
        )
        c5.metric(
            "Preemptions",
            compact_number(sim.preemptions, decimals=0),
            help="KV-pressure evictions. each victim resumes via recompute.",
        )
        c6.metric(
            "Dropped",
            compact_number(sim.dropped, decimals=0),
            help="Requests whose context can never fit a single replica.",
        )

        st.markdown("##### Latency percentiles")
        rows = []
        for key, label in (
            ("ttft_ms", "TTFT (ms)"),
            ("tpot_ms", "TPOT (ms)"),
            ("e2e_ms", "E2E  (ms)"),
        ):
            d = lat[key]
            rows.append(
                {
                    "metric": label,
                    "p50": d["p50"],
                    "p95": d["p95"],
                    "p99": d["p99"],
                    "mean": d["mean"],
                    "n": d["n"],
                }
            )
        st.dataframe(rows, use_container_width=True)

        if not sim.completed:
            st.warning(
                "No requests completed inside the window. The replica is likely "
                "overloaded (check Preemptions/Dropped) – lower lambda, add "
                "replicas, or shorten the workload."
            )
        else:
            ttfts = [r.ttft * 1000 for r in sim.completed if r.first_token_at >= 0]
            tpots = [r.tpot * 1000 for r in sim.completed if not math.isnan(r.tpot)]
            e2es = [r.e2e * 1000 for r in sim.completed if r.completed_at > 0]

            tab_lat, tab_rel, tab_ts, tab_rep = st.tabs(
                ["Latency", "Relationships", "Time series", "Replicas"]
            )

            with tab_lat:
                ttft_ok = (
                    float(np.mean([t <= slo.ttft_p95_ms for t in ttfts]))
                    if ttfts
                    else float("nan")
                )
                tpot_ok = (
                    float(np.mean([t <= slo.tpot_p95_ms for t in tpots]))
                    if tpots
                    else float("nan")
                )
                s1, s2 = st.columns(2)
                s1.metric("Meets TTFT SLO", f"{compact_number(ttft_ok * 100)}%")
                s2.metric("Meets TPOT SLO", f"{compact_number(tpot_ok * 100)}%")

                st.markdown("###### Latency CDFs (dashed red = p95 SLO)")
                cdf_a, cdf_b = st.columns(2)
                ch = cdf_chart(ttfts, "TTFT (ms)", slo.ttft_p95_ms)
                if ch is not None:
                    cdf_a.altair_chart(ch, use_container_width=True)
                ch = cdf_chart(tpots, "TPOT (ms)", slo.tpot_p95_ms)
                if ch is not None:
                    cdf_b.altair_chart(ch, use_container_width=True)
                ch = cdf_chart(e2es, "E2E (ms)")
                if ch is not None:
                    st.altair_chart(ch, width="stretch")

                with st.expander("Histograms"):
                    hcol1, hcol2 = st.columns(2)
                    for col, data, field, xl in (
                        (hcol1, ttfts, "ttft_ms", "TTFT bin center (ms)"),
                        (hcol2, tpots, "tpot_ms", "TPOT bin center (ms)"),
                    ):
                        if data:
                            hist, edges = np.histogram(data, bins=40)
                            centers = (edges[:-1] + edges[1:]) / 2
                            col.bar_chart(
                                [
                                    {field: round(float(c), 2), "requests": int(n_)}
                                    for c, n_ in zip(centers, hist, strict=True)
                                ],
                                x=field,
                                y="requests",
                                x_label=xl,
                                y_label="Completed requests",
                            )

            with tab_rel:
                st.caption(
                    "Each point is a completed request. Reveals how latency "
                    "scales with prompt length, context, and output length."
                )
                pts = [
                    {
                        "prompt_len": r.prompt_len,
                        "context": r.prompt_len + r.output_len,
                        "output_len": r.output_len,
                        "ttft_ms": r.ttft * 1000,
                        "tpot_ms": r.tpot * 1000,
                        "e2e_ms": r.e2e * 1000,
                    }
                    for r in sim.completed
                    if r.first_token_at >= 0 and r.completed_at > 0
                ]
                rcol1, rcol2 = st.columns(2)
                rcol1.markdown("###### TTFT vs prompt length")
                rcol1.scatter_chart(
                    pts,
                    x="prompt_len",
                    y="ttft_ms",
                    x_label="prompt tokens",
                    y_label="TTFT (ms)",
                )
                rcol2.markdown("###### TPOT vs context length")
                rcol2.scatter_chart(
                    pts,
                    x="context",
                    y="tpot_ms",
                    x_label="prompt+output tokens",
                    y_label="TPOT (ms)",
                )
                st.markdown("###### E2E vs output length")
                st.scatter_chart(
                    pts,
                    x="output_len",
                    y="e2e_ms",
                    x_label="output tokens",
                    y_label="E2E (ms)",
                )

            with tab_ts:
                if sim.mem_samples:
                    ts = np.array([t for t, _, _ in sim.mem_samples])
                    occ = np.array([o for _, o, _ in sim.mem_samples])
                    bsz = np.array([b for _, _, b in sim.mem_samples])
                    if len(ts) > 800:
                        idx = np.linspace(0, len(ts) - 1, 800).astype(int)
                        ts, occ, bsz = ts[idx], occ[idx], bsz[idx]

                    st.markdown(
                        "###### KV-cache occupancy (vertical ticks = preemptions)"
                    )
                    occ_df = pd.DataFrame({"time_s": ts, "occupancy": occ})
                    occ_line = (
                        alt.Chart(occ_df)
                        .mark_line(color="#4c78a8")
                        .encode(
                            x=alt.X("time_s:Q", title="simulation time (s)"),
                            y=alt.Y(
                                "occupancy:Q",
                                title="KV used / budget",
                                scale=alt.Scale(domain=[0, 1.05]),
                            ),
                        )
                    )
                    layers = [occ_line]
                    if sim.preempt_times:
                        pt = sim.preempt_times
                        if len(pt) > 400:
                            pt = list(
                                np.array(pt)[
                                    np.linspace(0, len(pt) - 1, 400).astype(int)
                                ]
                            )
                        layers.append(
                            alt.Chart(pd.DataFrame({"time_s": pt}))
                            .mark_rule(color="#e45756", opacity=0.25)
                            .encode(x="time_s:Q")
                        )
                    st.altair_chart(
                        alt.layer(*layers).properties(
                            height=260, autosize=FIT_AUTOSIZE, padding=CHART_PADDING
                        ),
                        width="stretch",
                    )

                    st.markdown("###### Running batch size over time")
                    st.line_chart(
                        pd.DataFrame({"time_s": ts, "batch_size": bsz}),
                        x="time_s",
                        y="batch_size",
                        x_label="simulation time (s)",
                        y_label="requests in batch (pooled)",
                    )

                st.markdown("###### RPS over time")
                completion_times = np.array(
                    [r.completed_at for r in sim.completed if r.completed_at >= 0]
                )
                bucket_edges = np.arange(0.0, w.sim_duration_s + 1.0, 1.0)
                completed_per_bucket, _ = np.histogram(
                    completion_times, bins=bucket_edges
                )
                st.line_chart(
                    [
                        {
                            "time_s": float(bucket_edges[i]),
                            "completed_rps": int(c),
                            "offered_rps": w.lambda_rps,
                        }
                        for i, c in enumerate(completed_per_bucket)
                    ],
                    x="time_s",
                    y=["completed_rps", "offered_rps"],
                    x_label="simulation time (s)",
                    y_label="requests per second",
                )

                if sim.queue_depth:
                    st.markdown("###### Queue depth over time (per-replica samples)")
                    qts = np.array([t for t, _ in sim.queue_depth])
                    qqs = np.array([q for _, q in sim.queue_depth])
                    if len(qts) > 500:
                        idx = np.linspace(0, len(qts) - 1, 500).astype(int)
                        qts, qqs = qts[idx], qqs[idx]
                    st.line_chart(
                        [
                            {"time_s": t, "queue_depth": q}
                            for t, q in zip(qts, qqs, strict=True)
                        ],
                        x="time_s",
                        y="queue_depth",
                        x_label="simulation time (s)",
                        y_label="queued requests per replica",
                    )

            with tab_rep:
                st.markdown("###### Per-replica load balance")
                completed_by_rep: dict[int, int] = {}
                for r in sim.completed:
                    completed_by_rep[r.replica_id] = (
                        completed_by_rep.get(r.replica_id, 0) + 1
                    )
                rep_rows = [
                    {
                        "replica": f"R{i}",
                        "completed": completed_by_rep.get(i, 0),
                        "busy_pct": round(busy * 100, 1),
                    }
                    for i, busy in enumerate(sim.replica_busy)
                ]
                bcol1, bcol2 = st.columns(2)
                bcol1.bar_chart(
                    rep_rows, x="replica", y="completed", y_label="completed requests"
                )
                bcol2.bar_chart(rep_rows, x="replica", y="busy_pct", y_label="busy %")

                st.markdown("###### Latency breakdown (component percentiles, ms)")
                st.caption(
                    "Approximate: queue wait + prefill + decode, each at its own "
                    "percentile (so segments need not sum to the total percentile)."
                )
                waits, prefills, decodes = [], [], []
                for r in sim.completed:
                    if r.first_token_at < 0 or r.completed_at <= 0:
                        continue
                    waits.append(max(0.0, r.admitted_at - r.arrived_at) * 1000)
                    prefills.append(max(0.0, r.first_token_at - r.admitted_at) * 1000)
                    decodes.append(max(0.0, r.completed_at - r.first_token_at) * 1000)
                bd_rows = []
                for pct in (50, 95, 99):
                    bd_rows.extend(
                        [
                            {
                                "percentile": f"p{pct}",
                                "component": "queue wait",
                                "ms": float(np.percentile(waits, pct)),
                            },
                            {
                                "percentile": f"p{pct}",
                                "component": "prefill",
                                "ms": float(np.percentile(prefills, pct)),
                            },
                            {
                                "percentile": f"p{pct}",
                                "component": "decode",
                                "ms": float(np.percentile(decodes, pct)),
                            },
                        ]
                    )
                st.bar_chart(
                    bd_rows, x="percentile", y="ms", color="component", stack=True
                )

    elif view == "Sweeps":
        st.markdown("##### Sweeps")
        st.caption(
            "Vary one knob across a range to find crossovers, knees, and the "
            "max sustainable load. Closed-form sweeps update live. the latency "
            "sweep runs the simulator and is gated behind a button."
        )

        lam_now = max(0.5, w.lambda_rps)
        lam_grid = list(np.linspace(0.5, max(2.0 * lam_now, 4.0), 30))

        st.markdown("###### Required GPUs vs arrival rate")
        rows = sweep_gpus_vs_lambda(m, g, p, e, w, int(target_batch), lam_grid)
        st.altair_chart(
            multiline_gpu_chart(
                rows,
                "lambda_rps",
                "arrival rate (req/s)",
                marker_x=w.lambda_rps,
                hline=float(p.total_gpus),
            ),
            width="stretch",
        )

        st.markdown("###### Required GPUs vs mean output length")
        out_grid = list(np.linspace(50, max(2.0 * w.mean_output, 200), 24))
        rows = sweep_gpus_vs_seqlen(
            m, g, p, e, w, int(target_batch), out_grid, vary="output"
        )
        st.altair_chart(
            multiline_gpu_chart(
                rows,
                "seq_len",
                "mean output tokens",
                marker_x=w.mean_output,
                hline=float(p.total_gpus),
            ),
            width="stretch",
        )

        st.markdown("###### Bottleneck map")
        lam_axis = list(np.linspace(0.5, max(2.0 * lam_now, 4.0), 10))
        out_axis = list(np.linspace(50, max(2.0 * w.mean_output, 200), 8))
        grid = sweep_bottleneck_grid(
            m, g, p, e, w, int(target_batch), lam_axis, out_axis
        )
        gdf = pd.DataFrame(grid)
        req_mid = float((gdf["required"].min() + gdf["required"].max()) / 2)
        heat = (
            alt.Chart(gdf)
            .mark_rect()
            .encode(
                x=alt.X("lambda_rps:O", title="arrival rate (req/s)"),
                y=alt.Y("mean_output:O", title="mean output tokens", sort="descending"),
                color=alt.Color(
                    "required:Q", title="GPUs", scale=alt.Scale(scheme="blues")
                ),
                tooltip=["lambda_rps", "mean_output", "required", "bottleneck"],
            )
        )
        text = heat.mark_text(baseline="middle", fontSize=9).encode(
            text=alt.Text("bottleneck:N"),
            color=alt.condition(
                alt.datum.required >= req_mid,
                alt.value("white"),
                alt.value("#1a1a1a"),
            ),
        )
        st.altair_chart(
            (heat + text).properties(
                height=360, autosize=FIT_AUTOSIZE, padding=CHART_PADDING
            ),
            width="stretch",
        )

        st.markdown("###### Latency & goodput vs offered load (simulated)")
        sweep_dur = min(w.sim_duration_s, 30.0)
        st.caption(
            f"Runs the simulator at each rate for {sweep_dur:.0f}s. "
            "Dashed red lines are the p95 SLOs."
        )
        if st.button("Run latency-vs-load sweep"):
            sim_lams = list(np.linspace(0.5, max(2.0 * lam_now, 4.0), 12))
            with st.spinner(f"Running {len(sim_lams)} simulations..."):
                lrows = sweep_latency_vs_load(
                    m, g, p, e, w, int(max_batch), sim_lams, sweep_dur
                )
            ldf = pd.DataFrame(lrows)

            lat_long = ldf.melt(
                id_vars=["lambda_rps"],
                value_vars=["ttft_p95_ms", "tpot_p95_ms"],
                var_name="metric",
                value_name="ms",
            )
            knee = (
                alt.Chart(lat_long)
                .mark_line(point=True)
                .encode(
                    x=alt.X("lambda_rps:Q", title="offered rate (req/s)"),
                    y=alt.Y("ms:Q", title="p95 latency (ms)"),
                    color=alt.Color("metric:N", title=None),
                )
            )
            slo_rules = (
                alt.Chart(
                    pd.DataFrame(
                        {
                            "y": [slo.ttft_p95_ms, slo.tpot_p95_ms],
                            "metric": ["ttft", "tpot"],
                        }
                    )
                )
                .mark_rule(color="#e45756", strokeDash=[5, 5])
                .encode(y="y:Q")
            )
            st.altair_chart(
                (knee + slo_rules).properties(
                    height=320, autosize=FIT_AUTOSIZE, padding=CHART_PADDING
                ),
                width="stretch",
            )

            good = ldf[["lambda_rps", "offered_rps", "goodput_rps"]].rename(
                columns={"lambda_rps": "offered (req/s)"}
            )
            st.line_chart(
                good,
                x="offered (req/s)",
                y=["offered_rps", "goodput_rps"],
                x_label="offered rate (req/s)",
                y_label="requests per second",
            )

    else:  # Glossary
        st.markdown("##### Formula glossary")
        st.caption(
            "Reference for the Estimate tab. These formulas are capacity floors: "
            "they assume steady average load and no queueing."
        )

        def formula_block(title: str, formulas: list[str]) -> None:
            st.markdown(f"**{title}**")
            for formula in formulas:
                st.latex(formula)

        symbol_rows = [
            ("N", "model parameters"),
            ("L", "transformer layers"),
            ("H", "attention heads"),
            (r"H_{kv}", "KV heads (GQA/MQA aware)"),
            (r"d_h", "head dimension"),
            ("b", "bytes per value"),
            ("P", "mean prompt tokens"),
            ("O", "mean output tokens"),
            (r"\lambda", "request rate (req/s)"),
            ("B", "assumed decode batch"),
            ("R", "replicas"),
            (r"G_r", "GPUs per replica (TP × PP)"),
            ("W", "model weight memory in bytes"),
            (r"K_{tok}", "KV-cache bytes per token"),
            (r"F_{pre}", "prefill FLOPs per request"),
            (r"F_{step}", "decode FLOPs per step"),
            (r"\bar{C}", "average decode context length"),
            (r"D_{tok}", "decode bytes per output token"),
            (r"G_{compute}", "GPU floor from compute"),
            (r"G_{bw}", "GPU floor from HBM bandwidth"),
            (r"G_{required}", "required throughput GPUs"),
            (r"K_{budget}", "KV budget per replica"),
            ("Q", "average in-flight requests"),
            (r"K_{active}", "active KV tokens"),
            (r"M_{resident}", "resident memory bytes"),
            (r"G_{memory}", "aggregate memory floor"),
            (r"M_{gpu}", "HBM bytes per GPU"),
            ("h", "usable HBM headroom"),
            (r"\eta", "parallelism efficiency"),
            (r"F_{gpu,peak}", "peak FLOPs/sec per GPU"),
            (r"BW_{gpu,peak}", "peak HBM bandwidth per GPU"),
        ]

        st.markdown("##### Model facts")
        c1, c2 = st.columns(2)
        with c1:
            formula_block(
                "Weight memory",
                [r"W = N \cdot b"],
            )
            st.caption("Raw model weight footprint at the selected dtype.")
        with c2:
            formula_block(
                "KV cache per token",
                [r"K_{tok} = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot b"],
            )
            st.caption("KV bytes produced per token. Factor 2 is for keys and values.")

        st.markdown("##### Prefill and decode compute")
        c1, c2 = st.columns(2)
        with c1:
            formula_block(
                "Prefill FLOPs per request",
                [r"F_{pre}(P) = 2NP + 4LP^2Hd_h"],
            )
            st.caption(
                "Dense forward pass plus quadratic attention over the prompt. "
                "Prefill is computed per average request, then scaled by RPS."
            )
        with c2:
            formula_block(
                "Decode FLOPs per step",
                [r"F_{step}(B) = 2NB"],
            )
            st.caption(
                "One decode step emits one token per request in the batch. "
                "The Estimate tab uses B=1 for aggregate per-request FLOPs."
            )

        st.markdown("##### Throughput floors")
        formula_block(
            "Compute floor",
            [
                r"F_{req/s} = \lambda \left(F_{pre}(P) + 2NO\right)",
                r"G_{compute}"
                r" = \frac{F_{req/s}}"
                r"{F_{gpu,peak} \cdot \mathrm{MFU}_{pre}"
                r" \cdot \eta}",
            ],
        )
        st.caption(
            "How many GPUs are needed if useful FLOPs/sec were the limiting resource. "
            "Decode's low real-world MFU is mostly a bandwidth effect, so decode "
            "bandwidth is handled in the bandwidth floor instead."
        )

        formula_block(
            "Bandwidth floor",
            [
                r"\bar{C} = P + \frac{O}{2}",
                r"D_{tok} = \frac{W}{B} + \bar{C} \cdot K_{tok}",
                r"G_{bw}"
                r" = \frac{\lambda O \cdot D_{tok}}"
                r"{BW_{gpu,peak} \cdot \mathrm{MBU} \cdot \eta}",
            ],
        )
        st.caption(
            "Decode reads model weights once per step and amortizes that read over "
            "the assumed decode batch. KV reads are per request, so they are not "
            "divided by batch."
        )

        formula_block(
            "Required throughput GPUs",
            [r"G_{required} = \left\lceil \max(G_{compute}, G_{bw}) \right\rceil"],
        )
        st.caption(
            "The larger floor determines the binding throughput resource. Memory "
            "residency is checked separately."
        )

        st.markdown("##### Memory and topology")
        c1, c2 = st.columns(2)
        with c1:
            formula_block(
                "Minimum GPUs per replica for weights",
                [
                    r"G_{r,min}"
                    r" = \left\lceil"
                    r"\frac{W}"
                    r"{M_{gpu} \cdot h}"
                    r"\right\rceil"
                ],
            )
            st.caption("A single replica must hold one sharded copy of the model.")
        with c2:
            formula_block(
                "KV budget per replica",
                [
                    r"K_{budget}"
                    r" = \frac{G_r \cdot M_{gpu} \cdot h - W}"
                    r"{K_{tok}}"
                ],
            )
            st.caption("How many active KV tokens fit after loading weights.")

        st.markdown("##### Little's law estimate for active KV")
        formula_block(
            "Active KV estimate",
            [
                r"T_{resp} \approx T_{pre} + O \cdot T_{step}",
                r"Q = \lambda \cdot T_{resp}",
                r"K_{act} = Q \cdot \bar{C}",
                r"K_{act,replica} = \frac{K_{act}}{R}",
            ],
        )
        st.caption(
            "This is an approximate no-queueing residency estimate. The simulator "
            "handles queueing, random lengths, preemption, and recompute."
        )

        st.markdown("##### Aggregate memory floor")
        formula_block(
            "Memory floor",
            [
                r"M_{res} = R \cdot W + K_{act} \cdot K_{tok}",
                r"G_{memory} = \frac{M_{res}}{M_{gpu} \cdot h}",
            ],
        )

        st.markdown("##### Efficiency knobs")
        st.markdown(
            """
            - **MFU (prefill)**: useful FLOPs as a fraction of peak during prefill.
            - **MFU (decode)**: useful FLOPs as a fraction of peak during decode timing.
            - **MBU (decode)**: useful HBM bandwidth as a fraction of peak during decode.
            - **Parallelism efficiency**: discount for TP/PP communication and imperfect
              multi-GPU scaling.
            - **HBM headroom**: fraction of HBM treated as usable after runtime overhead.
            """
        )

        st.markdown("##### Notation")
        notation_rows = "\n".join(
            f"| ${symbol}$ | {meaning} |" for symbol, meaning in symbol_rows
        )
        st.markdown(
            "| Symbol | Meaning |\n"
            "|---:|---|\n"
            f"{notation_rows}"
        )


def running_under_streamlit() -> bool:
    try:
        import streamlit.runtime as _sr

        return _sr.exists()
    except Exception:
        return False


if running_under_streamlit():
    run_ui()
