#!/usr/bin/env python3
"""Isolated MPS Metal experiment for Parakeet add + LayerNorm fusion.

This does not patch Kestrel. It compiles one process-local shader and compares it
with the torch reference used by Kestrel 0.8 on MPS.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
from typing import Callable

LOCK = Path("/tmp/ptt-mps-benchmark.lock")
WIDTH = 1024
GROUP_SIZE = 256

SHADER = r"""
#include <metal_stdlib>
using namespace metal;

// Parakeet Redux has C=1024. One threadgroup owns one [C] row.
kernel void add_scaled_layer_norm_1024(
    device half* total [[buffer(0)]],
    device half* normalized [[buffer(1)]],
    device const half* residual [[buffer(2)]],
    device const half* update [[buffer(3)]],
    device const half* weight [[buffer(4)]],
    device const half* bias [[buffer(5)]],
    constant float& alpha [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial_sum[8];
  threadgroup float partial_square[8];
  threadgroup float stats[2];
  const uint base = row * 1024;
  float sum = 0.0f;
  float square = 0.0f;

  for (uint channel = lane; channel < 1024; channel += 256) {
    const uint index = base + channel;
    // Round the residual sum to fp16 before normalization, like torch.add.
    half value = residual[index] + half(alpha) * update[index];
    total[index] = value;
    float value_f = float(value);
    sum += value_f;
    square += value_f * value_f;
  }
  sum = simd_sum(sum);
  square = simd_sum(square);
  if (simd_lane == 0) {
    partial_sum[simd_group] = sum;
    partial_square[simd_group] = square;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_group == 0) {
    float group_sum = simd_lane < 8 ? partial_sum[simd_lane] : 0.0f;
    float group_square = simd_lane < 8 ? partial_square[simd_lane] : 0.0f;
    group_sum = simd_sum(group_sum);
    group_square = simd_sum(group_square);
    if (simd_lane == 0) {
      float mean = group_sum * (1.0f / 1024.0f);
      float variance = max(
          group_square * (1.0f / 1024.0f) - mean * mean, 0.0f);
      stats[0] = mean;
      stats[1] = rsqrt(variance + 1.0e-5f);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float mean = stats[0];
  const float inverse_std = stats[1];
  for (uint channel = lane; channel < 1024; channel += 256) {
    const uint index = base + channel;
    float value = float(total[index]);
    normalized[index] = half(
        (value - mean) * inverse_std * float(weight[channel])
        + float(bias[channel]));
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[48, 80, 128, 224])
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def timed_us(torch, function: Callable[[], None], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.mps.synchronize()
    start = time.perf_counter_ns()
    for _ in range(iterations):
        function()
    torch.mps.synchronize()
    return (time.perf_counter_ns() - start) / iterations / 1_000.0


def run(args: argparse.Namespace) -> dict:
    import torch
    import torch.nn.functional as F

    if not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("this torch has no torch.mps.compile_shader")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    if any(row <= 0 for row in args.rows):
        raise ValueError("--rows values must be positive")
    if min(args.iterations, args.warmup, args.repeats) <= 0:
        raise ValueError("timing counts must be positive")

    library = torch.mps.compile_shader(SHADER)
    kernel = library.add_scaled_layer_norm_1024
    if kernel.thread_execution_width != 32 or kernel.max_threads_per_threadgroup < GROUP_SIZE:
        raise RuntimeError(
            "shader expects a 32-lane SIMD group and at least 256 threads per threadgroup"
        )

    results = []
    for rows in args.rows:
        torch.manual_seed(args.seed + rows)
        residual = torch.randn((rows, WIDTH), device="mps", dtype=torch.float16)
        update = torch.randn_like(residual)
        weight = torch.randn((WIDTH,), device="mps", dtype=torch.float16)
        bias = torch.randn_like(weight)
        total_out = torch.empty_like(residual)
        normalized_out = torch.empty_like(residual)

        def launch(alpha: float, total, normalized) -> None:
            kernel(
                total,
                normalized,
                residual,
                update,
                weight,
                bias,
                alpha,
                threads=GROUP_SIZE * rows,
                group_size=GROUP_SIZE,
            )

        correctness = []
        for alpha in (0.5, 1.0):
            launch(alpha, total_out, normalized_out)
            total_ref = torch.add(residual, update, alpha=alpha)
            normalized_ref = F.layer_norm(
                total_ref, (WIDTH,), weight, bias, 1.0e-5
            )
            torch.mps.synchronize()
            difference = (normalized_out - normalized_ref).abs().float()
            correctness.append(
                {
                    "alpha": alpha,
                    "total_max_abs": float((total_out - total_ref).abs().max()),
                    "normalized_max_abs": float(difference.max()),
                    "normalized_mean_abs": float(difference.mean()),
                    "normalized_exact_fraction": float(
                        (normalized_out == normalized_ref).float().mean()
                    ),
                    "finite": bool(torch.isfinite(normalized_out).all()),
                    "passes_tolerance": bool(
                        float((total_out - total_ref).abs().max()) == 0.0
                        and torch.isfinite(normalized_out).all()
                        and float(difference.mean()) <= 1.0e-6
                        and float(difference.max()) <= 1.0e-2
                    ),
                }
            )

        def metal_preallocated() -> None:
            launch(0.5, total_out, normalized_out)

        def metal_allocating() -> None:
            launch(0.5, torch.empty_like(residual), torch.empty_like(residual))

        def torch_eager() -> None:
            total = torch.add(residual, update, alpha=0.5)
            F.layer_norm(total, (WIDTH,), weight, bias, 1.0e-5)

        timings = {}
        for name, function in (
            ("metal_preallocated_us", metal_preallocated),
            ("metal_allocating_us", metal_allocating),
            ("torch_eager_us", torch_eager),
        ):
            samples = [
                timed_us(torch, function, args.warmup, args.iterations)
                for _ in range(args.repeats)
            ]
            timings[name] = statistics.median(samples)
            timings[name.replace("_us", "_samples_us")] = samples
        timings["speedup_allocating_over_eager"] = (
            timings["torch_eager_us"] / timings["metal_allocating_us"]
        )
        results.append({"rows": rows, "correctness": correctness, **timings})

    return {
        "experiment": "Parakeet fp16 add_scaled_layer_norm C=1024",
        "torch_version": torch.__version__,
        "device": "mps",
        "iterations": args.iterations,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "results": results,
    }


def main() -> None:
    args = parse_args()
    try:
        os.mkdir(LOCK)
    except FileExistsError as error:
        raise SystemExit(
            f"refusing to overlap an MPS benchmark; lock exists: {LOCK}"
        ) from error
    try:
        report = run(args)
        text = json.dumps(report, indent=2)
        if args.output is not None:
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
    finally:
        os.rmdir(LOCK)


if __name__ == "__main__":
    main()
