#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Profile the tiled Metal attention kernels with Instruments, from the terminal.

Records a Metal System Trace with GPU counters while the forward, dQ and dK/dV
kernels run in separate phases, then prints for each kernel:
  * pipeline limits (max threads per threadgroup, which falls as register use
    rises, and static threadgroup memory),
  * register spills reported by the shader compiler,
  * GPU time per call,
  * averaged performance-limiter counters (ALU, buffer reads, last-level
    cache, occupancy, bandwidth).

    python scripts/profile_mps_kernels.py
    python scripts/profile_mps_kernels.py --pattern window --seq 8192 --dtype float16

Requires Xcode (for xcrun xctrace) with its license accepted. The trace is
kept in --out for opening in Instruments.
"""

import argparse
import bisect
import collections
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

PHASES = ["forward", "dq", "dkdv"]
KERNELS = {"forward": "tiled_forward", "dq": "tiled_backward_dq", "dkdv": "tiled_backward_dkdv"}
COUNTERS = [
    "Compute Occupancy", "ALU Limiter", "ALU Utilization", "F32 Utilization", "F16 Utilization",
    "Buffer Read Limiter", "Buffer Write Limiter", "Threadgroup/Imageblock Load Limiter",
    "Threadgroup/Imageblock Store Limiter", "GPU Last Level Cache Limiter", "MMU Limiter",
    "GPU Read Bandwidth", "GPU Write Bandwidth",
]
PHASE_GAP_NS = 150_000_000   # the workload sleeps between phases
SETUP_GAP_S = 0.5
PHASE_SLEEP_S = 0.3


def pattern_args(pattern):
    from layers import mps_attention as M
    if pattern == "financial":
        params = [x for s in M.financial_segments(512, 1000, 8, 10) for x in s]
        return M.KIND_SEGMENTS, params, len(params) // 2
    if pattern == "window":
        return M.KIND_SEGMENTS, [-64, 64], 1
    if pattern == "block":
        return M.KIND_BLOCK_SPARSE, [64], 0
    return M.KIND_LONGFORMER, [64, 2], 0


def kernel_calls(args):
    """Build the three kernel launches on fresh inputs (workload and pipeline stats)."""
    import torch
    from layers import mps_attention as M
    dtype = getattr(torch, args.dtype)
    B, S, H, D = 1, args.seq, args.heads, args.head_dim
    S_pad = -(-S // M._TILE) * M._TILE
    q, k, v, g = (torch.randn(B, S_pad, H, D, device="mps").to(dtype) for _ in range(4))
    lib = M._tiled_library(D, M._METAL_TYPES[dtype])
    kind, params, nseg = pattern_args(args.pattern)
    pat = M._param_tensor(tuple(params), q.device)
    launch = M._tiled_launch(B, H, S_pad)
    scale = 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    lse = torch.empty(B, S_pad, H, device="mps")
    common = (pat, kind, nseg, B, S, S_pad, H, scale)
    lib.tiled_forward(q, k, v, out, lse, *common, **launch)
    delta = (g.float() * out.float()).sum(-1).contiguous()
    dq, dk, dv = (torch.empty_like(q) for _ in range(3))
    return lib, {
        "forward": lambda: lib.tiled_forward(q, k, v, out, lse, *common, **launch),
        "dq": lambda: lib.tiled_backward_dq(q, k, v, g, lse, delta, dq, *common, **launch),
        "dkdv": lambda: lib.tiled_backward_dkdv(q, k, v, g, lse, delta, dk, dv, *common, **launch),
    }


def run_workload(args):
    import torch
    _, calls = kernel_calls(args)
    torch.mps.synchronize()
    time.sleep(SETUP_GAP_S)
    for phase in PHASES:
        calls[phase]()
        for _ in range(args.reps):
            calls[phase]()
        torch.mps.synchronize()
        time.sleep(PHASE_SLEEP_S)


# --- Trace export and parsing ------------------------------------------------------

def export_table(trace, schema, path):
    xpath = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
    with open(path, "w") as f:
        subprocess.run(["xcrun", "xctrace", "export", "--input", trace, "--xpath", xpath],
                       stdout=f, check=True)
    return path


def table_rows(path):
    """Rows of an exported table as {column: element}, resolving id/ref de-duplication."""
    root = ET.parse(path).getroot()
    schema = root.find(".//schema")
    if schema is None:
        return []
    cols = [c.findtext("mnemonic") for c in schema.findall("col")]
    by_id = {el.attrib["id"]: el for el in root.iter() if "id" in el.attrib}
    out = []
    for row in root.iter("row"):
        rec = {}
        for col, el in zip(cols, list(row)):
            rec[col] = by_id[el.attrib["ref"]] if "ref" in el.attrib else el
        out.append(rec)
    return out


def counter_samples(path):
    """Stream the (large) counter table into {counter_id: sorted [(time_ns, value)]}."""
    by_id, samples = {}, collections.defaultdict(list)
    for _, el in ET.iterparse(path, events=("end",)):
        if "id" in el.attrib:
            by_id[el.attrib["id"]] = el.text
        if el.tag == "row":
            t, cid, value = (by_id[c.attrib["ref"]] if "ref" in c.attrib else c.text for c in list(el)[:3])
            samples[int(cid)].append((int(t), float(value)))
            el.clear()
    for values in samples.values():
        values.sort()
    return samples


def analyze(trace, work, reps):
    tables = {s: export_table(trace, s, os.path.join(work, f"{s}.xml")) for s in
              ("metal-gpu-intervals", "graphics-compiler-spill-events", "gpu-counter-info", "gpu-counter-value")}

    intervals = []
    for r in table_rows(tables["metal-gpu-intervals"]):
        if "python" in (r["process"].attrib.get("fmt") or "") and r["channel-name"].text == "Compute":
            start, duration = int(r["start"].text), int(r["duration"].text)
            intervals.append((start, start + duration, int(r["encoder-id"].text or 0)))
    intervals.sort()

    # Group intervals into phases by the sleeps between them; the last groups are the phases
    groups = []
    for iv in intervals:
        if groups and iv[0] - groups[-1][-1][1] < PHASE_GAP_NS:
            groups[-1].append(iv)
        else:
            groups.append([iv])
    if len(groups) < len(PHASES):
        sys.exit(f"found {len(groups)} GPU activity groups, expected at least {len(PHASES)}")
    phases = dict(zip(PHASES, groups[-len(PHASES):]))

    spills = collections.defaultdict(int)
    for r in table_rows(tables["graphics-compiler-spill-events"]):
        enc = int(r["encoder-id"].text)
        spills[enc] = max(spills[enc], int(r["spilled-bytes"].text))

    names = {int(r["counter-id"].text): r["name"].attrib.get("fmt") or r["name"].text
             for r in table_rows(tables["gpu-counter-info"])}
    samples = counter_samples(tables["gpu-counter-value"])

    result = {}
    for phase, spans in phases.items():
        busy = sum(b - a for a, b, _ in spans)
        counters = {}
        for cid, name in names.items():
            if name not in COUNTERS:
                continue
            times = [t for t, _ in samples.get(cid, [])]
            values = []
            for a, b, _ in spans:
                lo, hi = bisect.bisect_left(times, a), bisect.bisect_right(times, b)
                values += [v for _, v in samples[cid][lo:hi]]
            counters[name] = sum(values) / len(values) if values else float("nan")
        result[phase] = {
            "gpu_ms_per_call": busy / (reps + 1) / 1e6,
            "spilled_bytes": max((spills.get(enc, 0) for _, _, enc in spans), default=0),
            "counters": counters,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pattern", choices=["financial", "window", "block", "longformer"], default="financial")
    parser.add_argument("--seq", type=int, default=16384)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--out", help="directory for the trace and exported tables (default: a temp dir)")
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    parser.add_argument("--workload", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.workload:
        run_workload(args)
        return
    if shutil.which("xcrun") is None or subprocess.run(["xcrun", "--find", "xctrace"],
                                                       capture_output=True).returncode != 0:
        sys.exit("xctrace not found: install Xcode, accept its license, and select it with xcode-select")

    work = args.out or tempfile.mkdtemp(prefix="mps-profile-")
    os.makedirs(work, exist_ok=True)
    trace = os.path.join(work, "kernels.trace")
    shutil.rmtree(trace, ignore_errors=True)
    workload = [sys.executable, os.path.abspath(__file__), "--workload", "--pattern", args.pattern,
                "--seq", str(args.seq), "--heads", str(args.heads), "--head-dim", str(args.head_dim),
                "--dtype", args.dtype, "--reps", str(args.reps)]
    subprocess.run(["xcrun", "xctrace", "record", "--template", "Metal System Trace",
                    "--instrument", "Metal GPU Counters", "--output", trace, "--launch", "--", *workload],
                   check=True, stdout=subprocess.DEVNULL)

    lib, _ = kernel_calls(args)
    result = analyze(trace, work, args.reps)
    for phase in PHASES:
        kernel = getattr(lib, KERNELS[phase])
        result[phase]["max_threads_per_threadgroup"] = kernel.max_threads_per_threadgroup
        result[phase]["threadgroup_memory_bytes"] = kernel.static_thread_group_memory_length

    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"{args.pattern}, seq {args.seq}, {args.heads} heads, head dim {args.head_dim}, {args.dtype}"
          f"  (trace: {trace})\n")
    rows = [("GPU time per call (ms)", lambda r: f"{r['gpu_ms_per_call']:.2f}"),
            ("Max threads/threadgroup", lambda r: str(r["max_threads_per_threadgroup"])),
            ("Threadgroup memory (bytes)", lambda r: str(r["threadgroup_memory_bytes"])),
            ("Register spill (bytes/thread)", lambda r: str(r["spilled_bytes"]))]
    rows += [(name, lambda r, n=name: f"{r['counters'].get(n, float('nan')):.1f}") for name in COUNTERS]
    print(f"{'':38s}" + "".join(f"{p:>12s}" for p in PHASES))
    for label, fmt in rows:
        print(f"{label:38s}" + "".join(f"{fmt(result[p]):>12s}" for p in PHASES))


if __name__ == "__main__":
    main()
