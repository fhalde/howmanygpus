from __future__ import annotations

import math

import numpy as np
import simpy

from howmanygpus.sizing import (
    decode_flops_per_step,
    kv_budget_tokens,
    kv_bytes_per_token,
    prefill_flops,
    replica_bw_per_s,
    replica_flops_per_s,
    weights_bytes,
)
from howmanygpus.spec import (
    EfficiencySpec,
    GPUSpec,
    ModelSpec,
    ParallelismSpec,
    Request,
    SimResult,
    WorkloadSpec,
)


class Replica:
    def __init__(
        self,
        env: simpy.Environment,
        replica_id: int,
        model: ModelSpec,
        gpu: GPUSpec,
        parallelism: ParallelismSpec,
        eff: EfficiencySpec,
        max_batch: int,
        completed: list[Request],
    ):
        self.env = env
        self.id = replica_id
        self.m = model
        self.g = gpu
        self.p = parallelism
        self.e = eff
        self.max_batch = max_batch
        self.completed = completed

        self.kv_budget = kv_budget_tokens(model, gpu, parallelism, eff)
        self.queue: list[Request] = []
        self.batch: list[Request] = []
        self.kv_used = 0.0
        self.wake = env.event()
        self.busy_time = 0.0
        self.queue_depth_samples: list[tuple[float, int]] = []
        self.mem_samples: list[tuple[float, float, int]] = []
        self.preempt_times: list[float] = []
        self.preemptions = 0
        self.dropped = 0

        self.proc = env.process(self.driver())

    @property
    def load(self) -> int:
        return len(self.queue) + len(self.batch)

    def submit(self, req: Request) -> None:
        req.replica_id = self.id
        self.queue.append(req)
        if not self.wake.triggered:
            self.wake.succeed()

    @staticmethod
    def resume_context(req: Request) -> int:
        return req.prompt_len + max(1, req.tokens_done)

    def can_admit(self, req: Request) -> bool:
        # Optimistic admission: only require the KV that must be resident *now*
        # to (re)start the request. Growth is reclaimed via preemption.
        if len(self.batch) >= self.max_batch:
            return False
        return self.kv_used + self.resume_context(req) <= self.kv_budget

    def prefill_seconds(self, prompt_len: int) -> float:
        flops = prefill_flops(self.m, prompt_len)
        per_s = replica_flops_per_s(self.g, self.p, self.e.mfu_prefill, self.e)
        return flops / per_s if per_s > 0 else math.inf

    def decode_step_seconds(self) -> float:
        batch_size = len(self.batch)
        if batch_size == 0:
            return 0.0
        ctx_total = sum(r.kv_tokens for r in self.batch)
        bytes_step = weights_bytes(self.m) + ctx_total * kv_bytes_per_token(self.m)
        flops_step = decode_flops_per_step(self.m, batch_size)
        bw = replica_bw_per_s(self.g, self.p, self.e)
        comp = replica_flops_per_s(self.g, self.p, self.e.mfu_decode, self.e)
        t_bw = bytes_step / bw if bw > 0 else math.inf
        t_comp = flops_step / comp if comp > 0 else math.inf
        return max(t_bw, t_comp)

    def driver(self):
        while True:
            self.queue_depth_samples.append((self.env.now, len(self.queue)))
            self.mem_samples.append(
                (
                    self.env.now,
                    self.kv_used / max(1.0, self.kv_budget),
                    len(self.batch),
                )
            )

            # Admit at most one request per cycle, then always run a decode step
            # so prefill cannot starve the in-flight batch.
            if self.queue and self.can_admit(self.queue[0]):
                req = self.queue.pop(0)
                is_fresh = req.tokens_done == 0
                ctx_len = (
                    req.prompt_len if is_fresh else req.prompt_len + req.tokens_done
                )
                req.admitted_at = self.env.now
                t_pf = self.prefill_seconds(ctx_len)
                yield self.env.timeout(t_pf)
                self.busy_time += t_pf
                if is_fresh:
                    req.first_token_at = self.env.now
                    req.tokens_done = 1
                req.kv_tokens = req.prompt_len + req.tokens_done
                if req.tokens_done >= req.output_len:
                    req.completed_at = self.env.now
                    self.completed.append(req)
                else:
                    self.kv_used += req.kv_tokens
                    self.batch.append(req)

            if not self.batch:
                if self.queue:
                    self.queue.pop(0)
                    self.dropped += 1
                    continue
                yield self.wake
                self.wake = self.env.event()
                continue

            t_step = self.decode_step_seconds()
            yield self.env.timeout(t_step)
            self.busy_time += t_step

            done_now: list[Request] = []
            for r in self.batch:
                r.tokens_done += 1
                r.kv_tokens += 1
                self.kv_used += 1
                if r.tokens_done >= r.output_len:
                    r.completed_at = self.env.now
                    done_now.append(r)

            for r in done_now:
                self.batch.remove(r)
                self.kv_used -= r.kv_tokens
                self.completed.append(r)

            # KV pressure: preempt newest-first (vLLM-style). The victim drops
            # KV and recomputes context on resume while keeping generated tokens.
            while self.kv_used > self.kv_budget and self.batch:
                victim = max(self.batch, key=lambda r: r.arrived_at)
                self.batch.remove(victim)
                self.kv_used -= victim.kv_tokens
                victim.kv_tokens = 0
                self.queue.insert(0, victim)
                self.preemptions += 1
                self.preempt_times.append(self.env.now)


def lognormal_sample(rng: np.random.Generator, mean: float, cv: float) -> float:
    if cv <= 0:
        return float(mean)
    sigma2 = math.log(1.0 + cv * cv)
    sigma = math.sqrt(sigma2)
    mu = math.log(max(1e-9, mean)) - 0.5 * sigma2
    return float(rng.lognormal(mu, sigma))


def arrival_proc(
    env: simpy.Environment,
    replicas: list[Replica],
    w: WorkloadSpec,
    rng: np.random.Generator,
):
    rid = 0
    while True:
        dt = rng.exponential(1.0 / max(1e-9, w.lambda_rps))
        yield env.timeout(dt)
        if env.now > w.sim_duration_s:
            return
        prompt_len = max(1, int(lognormal_sample(rng, w.mean_prompt, w.cv_prompt)))
        output_len = max(1, int(lognormal_sample(rng, w.mean_output, w.cv_output)))
        target = min(replicas, key=lambda r: r.load)
        target.submit(
            Request(
                id=rid,
                arrived_at=env.now,
                prompt_len=prompt_len,
                output_len=output_len,
            )
        )
        rid += 1


def simulate(
    m: ModelSpec,
    g: GPUSpec,
    p: ParallelismSpec,
    e: EfficiencySpec,
    w: WorkloadSpec,
    max_batch: int = 256,
) -> SimResult:
    env = simpy.Environment()
    completed: list[Request] = []
    replicas = [
        Replica(env, i, m, g, p, e, max_batch, completed) for i in range(p.replicas)
    ]
    rng = np.random.default_rng(w.seed)
    env.process(arrival_proc(env, replicas, w, rng))
    env.run(until=w.sim_duration_s + 30.0)

    busy = sum(min(r.busy_time, w.sim_duration_s) for r in replicas)
    capacity = w.sim_duration_s * p.replicas
    util = busy / capacity if capacity > 0 else 0.0

    qd = []
    mem = []
    pre_times = []
    for r in replicas:
        qd.extend(r.queue_depth_samples)
        mem.extend(r.mem_samples)
        pre_times.extend(r.preempt_times)
    qd.sort()
    mem.sort()
    pre_times.sort()
    replica_busy = [
        min(r.busy_time, w.sim_duration_s) / w.sim_duration_s for r in replicas
    ]

    return SimResult(
        completed=completed,
        queue_depth=qd,
        utilization=util,
        sim_time_s=w.sim_duration_s,
        preemptions=sum(r.preemptions for r in replicas),
        dropped=sum(r.dropped for r in replicas),
        mem_samples=mem,
        preempt_times=pre_times,
        replica_busy=replica_busy,
    )
