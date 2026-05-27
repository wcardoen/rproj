"""
check_mantis_bench.py
=====================
Runs Mantis inference on both GPU (CUDA) and CPU, then reports:
  - Wall-time for each device
  - GPU memory allocated / reserved (GPU run only)
  - Numerical difference between the two sets of predictions

Reproducibility
---------------
set_seed() is called once at the start of main().  It seeds Python,
NumPy, and all PyTorch RNGs and enables deterministic cuDNN mode so
that repeated runs on the same device produce bit-identical results.
Note: GPU and CPU results will still differ slightly due to differing
floating-point operation order (see comparison section at the end).
"""

import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

import mantis

# ------------------------------------------------------------------ #
# Configuration — edit these to match your environment                #
# ------------------------------------------------------------------ #
INP_FILE    = "../data/example.csv"
MODEL_DIR   = "../models"
HORIZON     = 4
USE_COV     = False
TARGET_TYPE = 1
COVARIATE   = None
COV_TYPE    = None
N_WARMUP    = 1   # warm-up passes before timing (GPU JIT / cuDNN warm-up)
N_RUNS      = 5   # timed passes (wall-time is averaged over these)
SEED        = 42  # master seed for reproducibility
# ------------------------------------------------------------------ #


# ═══════════════════════════════════════════════════════════════════ #
#  Reproducibility                                                    #
# ═══════════════════════════════════════════════════════════════════ #

def set_seed(seed: int = SEED) -> None:
    """
    Seed every RNG that PyTorch and its dependencies expose so that
    repeated runs on the same device produce identical results.

    What each line does
    -------------------
    random.seed          : Python built-in RNG (used by some data utils)
    np.random.seed       : NumPy RNG (used in preprocessing)
    torch.manual_seed    : CPU RNG for all torch ops
    cuda.manual_seed_all : RNG on every GPU, not just GPU 0
    cudnn.deterministic  : forces cuDNN to pick the same algorithm
                           every run (default: fastest, which can vary)
    cudnn.benchmark=False: disables the auto-tuner that benchmarks
                           several cuDNN kernels at start-up — the
                           'winner' can differ between runs
    use_deterministic_algorithms(True)
                         : raises RuntimeError if any op would use a
                           non-deterministic CUDA kernel, so nothing
                           slips through silently.

    Performance note
    ----------------
    cudnn.deterministic=True disables some fused kernels; expect a
    ~5-15 % slower GPU inference compared to the default settings.

    Caveat
    ------
    Seeding does NOT make GPU == CPU results.  It makes each device
    reproduce the *same* result on every run.  GPU/CPU divergence is a
    separate floating-point issue (parallel reduction order) unrelated
    to seeding.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)           # covers multi-GPU
        torch.backends.cudnn.deterministic = True  # same algorithm every run
        torch.backends.cudnn.benchmark     = False # disable auto-tuner

    # Loudly fail if a non-deterministic op sneaks in
    torch.use_deterministic_algorithms(True)


# ═══════════════════════════════════════════════════════════════════ #
#  Helpers                                                            #
# ═══════════════════════════════════════════════════════════════════ #

def gpu_stats(device: torch.device) -> dict:
    """Return current GPU memory stats (MB) for *device*."""
    if device.type != "cuda":
        return {}
    idx = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "allocated_MB" : torch.cuda.memory_allocated(idx)  / 1024**2,
        "reserved_MB"  : torch.cuda.memory_reserved(idx)   / 1024**2,
        "max_alloc_MB" : torch.cuda.max_memory_allocated(idx) / 1024**2,
    }


def run_inference(model: mantis.Mantis, ts: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Run one timed inference pass.

    For CUDA devices the timing uses CUDA events so it measures only GPU
    compute time (not Python overhead).  For CPU it uses time.perf_counter.

    Returns (predictions_array, wall_seconds).
    """
    device = model.device

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        start_evt.record()

        preds = model.predict(
            time_series    = ts,
            covariate      = COVARIATE,
            target_type    = TARGET_TYPE,
            covariate_type = COV_TYPE,
        )

        end_evt.record()
        torch.cuda.synchronize(device)
        elapsed = start_evt.elapsed_time(end_evt) / 1000.0   # ms → s
    else:
        t0    = time.perf_counter()
        preds = model.predict(
            time_series    = ts,
            covariate      = COVARIATE,
            target_type    = TARGET_TYPE,
            covariate_type = COV_TYPE,
        )
        elapsed = time.perf_counter() - t0

    return preds, elapsed


def bench(device_str: str, ts: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """
    Load the model on *device_str*, warm up, then time N_RUNS passes.

    Returns (mean_predictions, mean_wall_time_s, gpu_memory_dict).
    """
    print(f"\n{'='*60}")
    print(f"  Device : {device_str.upper()}")
    print(f"{'='*60}")

    # ── load model ────────────────────────────────────────────────
    t_load = time.perf_counter()
    model  = mantis.Mantis(
        forecast_horizon = HORIZON,
        use_covariate    = USE_COV,
        model_dir        = MODEL_DIR,
        device           = device_str,
    )
    print(f"  Model loaded in {time.perf_counter() - t_load:.3f} s  "
          f"(device resolved → {model.device})")

    # ── reset GPU stats before warm-up ────────────────────────────
    if model.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(model.device)

    # ── warm-up passes ────────────────────────────────────────────
    print(f"  Warm-up : {N_WARMUP} pass(es) …", end=" ", flush=True)
    for _ in range(N_WARMUP):
        run_inference(model, ts)
    print("done")

    # ── capture GPU memory AFTER warm-up, BEFORE timed runs ───────
    mem_before = gpu_stats(model.device)

    # ── timed passes ──────────────────────────────────────────────
    times  = []
    preds_all = []
    print(f"  Timing  : {N_RUNS} run(s) …")
    for i in range(N_RUNS):
        p, t = run_inference(model, ts)
        times.append(t)
        preds_all.append(p)
        print(f"    run {i+1}/{N_RUNS}  →  {t*1000:.2f} ms")

    mean_preds = np.mean(preds_all, axis=0)
    mean_time  = float(np.mean(times))
    std_time   = float(np.std(times))

    # ── GPU memory AFTER timed runs ───────────────────────────────
    mem_after = gpu_stats(model.device)

    # ── summary ───────────────────────────────────────────────────
    print(f"\n  ── Timing summary ──────────────────────────────────")
    print(f"  Mean wall-time : {mean_time*1000:.3f} ms  (±{std_time*1000:.3f} ms)")
    print(f"  Min            : {min(times)*1000:.3f} ms")
    print(f"  Max            : {max(times)*1000:.3f} ms")

    if mem_after:
        print(f"\n  ── GPU memory ──────────────────────────────────────")
        print(f"  Allocated (current) : {mem_after['allocated_MB']:.2f} MB")
        print(f"  Reserved  (current) : {mem_after['reserved_MB']:.2f} MB")
        print(f"  Peak allocated      : {mem_after['max_alloc_MB']:.2f} MB")

    return mean_preds, mean_time, mem_after


# ═══════════════════════════════════════════════════════════════════ #
#  Main                                                               #
# ═══════════════════════════════════════════════════════════════════ #

def main() -> None:

    # ── reproducibility (must be first) ───────────────────────────
    set_seed(SEED)

    # ── header ────────────────────────────────────────────────────
    print("Mantis GPU vs CPU benchmark")
    print(f"  mantis version : {mantis.__version__}")
    print(f"  torch version  : {torch.__version__}")
    print(f"  seed           : {SEED}")
    cuda_avail = torch.cuda.is_available()
    print(f"  CUDA available : {cuda_avail}")
    if cuda_avail:
        print(f"  CUDA device    : {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version   : {torch.version.cuda}")

    # ── load data ─────────────────────────────────────────────────
    print(f"\nReading '{INP_FILE}' …")
    try:
        df = pd.read_csv(INP_FILE)
    except FileNotFoundError:
        sys.exit(f"ERROR: input file not found: {INP_FILE}")
    ts = np.array(df["hospitalizationCount"])
    print(f"  time series shape : {ts.shape}")
    print(f"  first 5 values    : {ts[:5]}")

    # ── GPU run ───────────────────────────────────────────────────
    gpu_preds, gpu_time, gpu_mem = None, None, {}
    if cuda_avail:
        gpu_preds, gpu_time, gpu_mem = bench("cuda", ts)
    else:
        print("\n[SKIP] CUDA not available — skipping GPU run.")

    # ── CPU run ───────────────────────────────────────────────────
    cpu_preds, cpu_time, _ = bench("cpu", ts)

    # ── comparison ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")

    if gpu_preds is not None and cpu_preds is not None:
        abs_diff  = np.abs(gpu_preds - cpu_preds)
        rel_diff  = abs_diff / (np.abs(cpu_preds) + 1e-8)

        print(f"\n  Predictions (GPU):\n{gpu_preds}\n")
        print(f"  Predictions (CPU):\n{cpu_preds}\n")
        print(f"  Absolute difference:")
        print(f"    max  : {abs_diff.max():.6f}")
        print(f"    mean : {abs_diff.mean():.6f}")
        print(f"  Relative difference:")
        print(f"    max  : {rel_diff.max():.6f}")
        print(f"    mean : {rel_diff.mean():.6f}")
        print(f"  torch.allclose (atol=1e-3) : "
              f"{np.allclose(gpu_preds, cpu_preds, atol=1e-3)}")
        speedup = cpu_time / gpu_time if gpu_time and gpu_time > 0 else float("nan")
        print(f"\n  Wall-time  GPU  : {gpu_time*1000:.3f} ms")
        print(f"  Wall-time  CPU  : {cpu_time*1000:.3f} ms")
        print(f"  Speed-up (CPU/GPU) : {speedup:.2f}×")
    else:
        print(f"  Predictions (CPU):\n{cpu_preds}\n")
        print(f"  Wall-time CPU : {cpu_time*1000:.3f} ms")

    print(f"\nDONE!")


if __name__ == "__main__":
    main()
