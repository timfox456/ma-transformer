#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark sparse attention on Apple silicon: vectorized PyTorch vs the Metal
kernels (per-row and tiled), forward and forward+backward.

    python scripts/benchmark_mps_attention.py
    python scripts/benchmark_mps_attention.py --quick --dtype float16
    python scripts/benchmark_mps_attention.py --markdown   # README table for this Mac

Times are the median of --iterations runs after one warm-up call. Before
printing the README table, --markdown checks the Metal kernels against the
PyTorch implementation on this GPU and refuses to print if they disagree.

The per-row kernel column is blank for block-sparse and Longformer, which only
the tiled kernels implement.
"""

import argparse
import os
import platform
import statistics
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from layers import blocked_attention, mps_attention  # noqa: E402

CASES = [
    # name, (batch, seq, heads, head_dim), pattern
    ("window w=16", (4, 4096, 1, 64), "window16"),
    ("window w=64", (1, 16384, 4, 64), "window64"),
    ("financial", (1, 16384, 4, 64), "financial"),
    ("block b=64", (1, 16384, 4, 64), "block"),
    ("longformer", (1, 16384, 4, 64), "longformer"),
    ("financial", (1, 65536, 4, 64), "financial"),
]
QUICK_CASES = CASES[:-1]
DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
TILED_ONLY = {"block", "longformer"}


def run(pattern, impl, q):
    module = blocked_attention if impl == "pytorch" else mps_attention
    if pattern == "block":
        return module.block_sparse_attention(q, q, q, 64)
    if pattern == "longformer":
        return module.longformer_attention(q, q, q, 64, 2)
    kwargs = {} if impl == "pytorch" else {"kernel": impl}
    if pattern == "financial":
        return module.financial_attention(q, q, q, **kwargs)
    window = 16 if pattern == "window16" else 64
    return module.sliding_window_attention(q, q, q, window, **kwargs)


def time_ms(fn, iterations):
    fn()
    torch.mps.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        torch.mps.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def machine_name():
    """e.g. "Apple M1 Pro (14-core GPU)"."""
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    profile = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True).stdout
    cores = next((line.split(":")[1].strip() for line in profile.splitlines() if "Total Number of Cores" in line), None)
    name = chip or platform.machine()
    return f"{name} ({cores}-core GPU)" if cores else name


# Rows of the README table: (label, sequence, case index in CASES)
MARKDOWN_ROWS = [
    ("Sliding window, w=64", 1),
    ("Financial (defaults)", 2),
    ("Block-sparse, b=64", 3),
    ("Longformer, w=64, 2 global", 4),
    ("Financial (defaults)", 5),
]


def format_ms(ms):
    """Whole milliseconds, with one decimal below 10 ms where rounding would hide differences."""
    return f"{ms:.1f} ms" if ms < 10 else f"{ms:,.0f} ms"


# Relative tolerance for the pre-benchmark check, by dtype (low precision rounds outputs)
CHECK_TOLERANCE = {"float32": 1e-4, "float16": 5e-3, "bfloat16": 3e-2}


def verify_kernels(args):
    """
    Compare the Metal kernels with the PyTorch implementation (forward and
    gradients) for every pattern in the table, at a size where the financial
    clusters are reached. Returns a list of failures, empty if all agree.
    """
    torch.manual_seed(0)
    dtype = DTYPES[args.dtype]
    shape = (1, 2100, 2, 64)
    failures = []
    for pattern in ("window64", "financial", "block", "longformer"):
        q, k, v = (torch.randn(shape, device="mps", dtype=dtype, requires_grad=True) for _ in range(3))
        grad_out = torch.randn(shape, device="mps", dtype=dtype)
        results = {}
        for impl in ("tiled", "pytorch"):
            inputs = [t if impl == "tiled" else t.detach().float().requires_grad_() for t in (q, k, v)]
            module = mps_attention if impl == "tiled" else blocked_attention
            if pattern == "block":
                out = module.block_sparse_attention(*inputs, 64)
            elif pattern == "longformer":
                out = module.longformer_attention(*inputs, 64, 2)
            elif pattern == "financial":
                out = module.financial_attention(*inputs)
            else:
                out = module.sliding_window_attention(*inputs, 64)
            grads = torch.autograd.grad(out, inputs, grad_out.to(out.dtype))
            results[impl] = [t.detach().float() for t in (out, *grads)]
        for name, got, want in zip(("output", "dQ", "dK", "dV"), results["tiled"], results["pytorch"]):
            err = (got - want).abs().max().item() / max(want.abs().max().item(), 1.0)
            if not err <= CHECK_TOLERANCE[args.dtype]:
                failures.append(f"{pattern} {name}: relative error {err:.2e}")
    return failures


def markdown_table(args):
    """Forward + backward times as the README's Markdown table."""
    failures = verify_kernels(args)
    if failures:
        sys.exit("Metal kernels disagree with the PyTorch implementation on this GPU; "
                 "not printing benchmark numbers:\n  " + "\n  ".join(failures))
    lines = [f"Forward + backward on {machine_name()}, batch 1, 4 heads, head dim 64, {args.dtype}:", "",
             "| Pattern | Sequence | PyTorch (vectorized) | Metal (tiled) |", "|---|---|---|---|"]
    for label, index in MARKDOWN_ROWS:
        _, shape, pattern = CASES[index]
        q = torch.randn(shape, device="mps", dtype=DTYPES[args.dtype], requires_grad=True)
        cells = [time_ms(lambda: run(pattern, impl, q).sum().backward(), args.iterations) for impl in ("pytorch", "tiled")]
        lines.append(f"| {label} | {shape[1]:,} | " + " | ".join(format_ms(ms) for ms in cells) + " |")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the 64K-token case")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--markdown", action="store_true", help="print the README table (forward + backward)")
    args = parser.parse_args()

    if not mps_attention.is_available():
        sys.exit("MPS kernels are not available on this machine")
    if args.markdown:
        markdown_table(args)
        return

    impls = ["pytorch", "row", "tiled"]
    header = f"{'case':12s} {'shape':20s}" + "".join(f" | {name + ' fwd / fwd+bwd (ms)':>28s}" for name in impls)
    print(header)
    print("-" * len(header))
    for name, shape, pattern in (QUICK_CASES if args.quick else CASES):
        q = torch.randn(shape, device="mps", dtype=DTYPES[args.dtype], requires_grad=True)
        row = f"{name:12s} {str(shape):20s}"
        for impl in impls:
            if impl == "row" and pattern in TILED_ONLY:
                row += f" | {'':>26s}"
                continue
            forward = time_ms(lambda: run(pattern, impl, q), args.iterations)
            both = time_ms(lambda: run(pattern, impl, q).sum().backward(), args.iterations)
            row += f" | {forward:12.1f} / {both:11.1f}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
