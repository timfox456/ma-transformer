#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Benchmark sparse attention on Apple silicon: vectorized PyTorch vs the Metal
kernels (per-row and tiled), forward and forward+backward.

    python scripts/benchmark_mps_attention.py
    python scripts/benchmark_mps_attention.py --quick
"""

import argparse
import os
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
    ("financial", (1, 65536, 4, 64), "financial"),
]
QUICK_CASES = CASES[:3]


def run(pattern, impl, q):
    if impl == "pytorch":
        module, kwargs = blocked_attention, {}
    else:
        module, kwargs = mps_attention, {"kernel": impl}
    if pattern == "financial":
        return module.financial_attention(q, q, q, **kwargs)
    window = 16 if pattern == "window16" else 64
    return module.sliding_window_attention(q, q, q, window, **kwargs)


def time_ms(fn, iterations):
    fn()
    torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.mps.synchronize()
    return (time.perf_counter() - start) / iterations * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the 64K-token case")
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    if not mps_attention.is_available():
        sys.exit("MPS kernels are not available on this machine")

    impls = ["pytorch", "row", "tiled"]
    header = f"{'case':12s} {'shape':20s}" + "".join(f" | {name + ' fwd / fwd+bwd (ms)':>28s}" for name in impls)
    print(header)
    print("-" * len(header))
    for name, shape, pattern in (QUICK_CASES if args.quick else CASES):
        q = torch.randn(shape, device="mps", requires_grad=True)
        row = f"{name:12s} {str(shape):20s}"
        for impl in impls:
            forward = time_ms(lambda: run(pattern, impl, q), args.iterations)
            both = time_ms(lambda: run(pattern, impl, q).sum().backward(), args.iterations)
            row += f" | {forward:12.1f} / {both:11.1f}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
