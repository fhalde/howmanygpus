from howmanygpus.spec import GPUSpec, ModelSpec

MODEL_PRESETS: dict[str, ModelSpec] = {
    "Llama-3-8B": ModelSpec(
        "Llama-3-8B", N=8.0e9, L=32, d_model=4096, n_heads=32, n_kv_heads=8, d_head=128
    ),
    "Llama-3-70B": ModelSpec(
        "Llama-3-70B",
        N=70.0e9,
        L=80,
        d_model=8192,
        n_heads=64,
        n_kv_heads=8,
        d_head=128,
    ),
    "Llama-3-405B": ModelSpec(
        "Llama-3-405B",
        N=405.0e9,
        L=126,
        d_model=16384,
        n_heads=128,
        n_kv_heads=8,
        d_head=128,
    ),
    "Mistral-7B": ModelSpec(
        "Mistral-7B", N=7.3e9, L=32, d_model=4096, n_heads=32, n_kv_heads=8, d_head=128
    ),
    "Qwen2.5-72B": ModelSpec(
        "Qwen2.5-72B",
        N=72.0e9,
        L=80,
        d_model=8192,
        n_heads=64,
        n_kv_heads=8,
        d_head=128,
    ),
}

GPU_PRESETS: dict[str, GPUSpec] = {
    "A100-80GB SXM": GPUSpec(
        "A100-80GB", flops_per_s=312e12, hbm_bytes_per_s=2.039e12, hbm_bytes=80e9
    ),
    "H100-80GB SXM": GPUSpec(
        "H100-80GB", flops_per_s=989e12, hbm_bytes_per_s=3.35e12, hbm_bytes=80e9
    ),
    "H200-141GB": GPUSpec(
        "H200-141GB", flops_per_s=989e12, hbm_bytes_per_s=4.8e12, hbm_bytes=141e9
    ),
    "B200": GPUSpec(
        "B200", flops_per_s=2250e12, hbm_bytes_per_s=8.0e12, hbm_bytes=192e9
    ),
    "MI300X": GPUSpec(
        "MI300X", flops_per_s=1307e12, hbm_bytes_per_s=5.3e12, hbm_bytes=192e9
    ),
}
