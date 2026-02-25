#!/usr/bin/env python3
"""Dump kernel and memcpy events from a time window to CSV.

Usage:
    python scripts/dump_window_csv.py --sqlite nsys_reports/*.sqlite --t1 2260 --t2 2400
    python scripts/dump_window_csv.py --sqlite nsys_reports/*.sqlite --t1 2260 --t2 2400 --out-dir selected_csv
"""
import argparse
import csv
import sqlite3
import os
import sys
from pathlib import Path

COPY_KIND = {0: "Unknown", 1: "HtoD", 2: "DtoH", 3: "HtoH", 4: "DtoD", 8: "Peer"}
MEM_KIND = {0: "Unknown", 1: "Pageable", 2: "Device", 3: "Array", 4: "Unified", 5: "Managed"}

KERNEL_SQL = """
SELECT
    k.start, k.end, k.end - k.start AS duration,
    k.deviceId, k.streamId,
    k.gridX, k.gridY, k.gridZ,
    k.blockX, k.blockY, k.blockZ,
    k.registersPerThread,
    k.staticSharedMemory + k.dynamicSharedMemory AS sharedMem,
    s.value AS demangledName
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
LEFT JOIN StringIds AS s ON k.demangledName = s.id
WHERE k.end >= {lo} AND k.start <= {hi}
ORDER BY k.start
"""

MEMCPY_SQL = """
SELECT
    m.start, m.end, m.end - m.start AS duration,
    m.bytes, m.copyKind, m.srcKind, m.dstKind,
    m.deviceId, m.streamId, m.correlationId
FROM CUPTI_ACTIVITY_KIND_MEMCPY AS m
WHERE m.end >= {lo} AND m.start <= {hi}
ORDER BY m.start
"""

MEMCPY_E2E_SQL = """
SELECT
    r.start AS cpu_start, r.end AS cpu_end,
    m.start AS gpu_start, m.end AS gpu_end,
    m.bytes, m.copyKind, m.srcKind, m.dstKind,
    m.deviceId, m.streamId,
    s.value AS api_name
FROM CUPTI_ACTIVITY_KIND_MEMCPY AS m
JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r ON m.correlationId = r.correlationId
LEFT JOIN StringIds AS s ON r.nameId = s.id
WHERE m.end >= {lo} AND m.start <= {hi}
ORDER BY m.start
"""


def fmt_bytes(b):
    if b >= 1 << 20:
        return f"{b / (1 << 20):.1f} MiB"
    if b >= 1 << 10:
        return f"{b / (1 << 10):.1f} KiB"
    return f"{b} B"


def short_name(demangled):
    if not demangled:
        return "unknown"
    name = demangled.split("<")[0].split("(")[0]
    parts = name.split("::")
    return "::".join(parts[-2:]) if len(parts) > 1 else parts[-1]


def main():
    p = argparse.ArgumentParser(description="Dump events from a time window to CSV")
    p.add_argument("--sqlite", required=True, help="Path to .sqlite file")
    p.add_argument("--t1", type=float, required=True, help="Window start (ms from trace start)")
    p.add_argument("--t2", type=float, required=True, help="Window end (ms from trace start)")
    p.add_argument("--out-dir", default="selected_csv", help="Output directory (default: selected_csv)")
    args = p.parse_args()

    conn = sqlite3.connect(args.sqlite)
    conn.row_factory = sqlite3.Row

    t0 = conn.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    if t0 is None:
        t0 = conn.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchone()[0]

    lo = t0 + int(args.t1 * 1e6)
    hi = t0 + int(args.t2 * 1e6)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{int(args.t1)}-{int(args.t2)}ms"

    # ── Kernels ──────────────────────────────────────────────────────────
    kernels = conn.execute(KERNEL_SQL.format(lo=lo, hi=hi)).fetchall()
    kernel_csv = os.path.join(args.out_dir, f"kernels_{tag}.csv")
    with open(kernel_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "start_ms", "end_ms", "duration_ms",
            "device_id", "stream_id",
            "grid", "block", "registers", "shared_mem_bytes",
            "op_type", "op_name", "full_name",
        ])
        for r in kernels:
            grid = f"{r['gridX']}x{r['gridY']}x{r['gridZ']}"
            block = f"{r['blockX']}x{r['blockY']}x{r['blockZ']}"
            w.writerow([
                f"{(r['start'] - t0) / 1e6:.4f}",
                f"{(r['end'] - t0) / 1e6:.4f}",
                f"{r['duration'] / 1e6:.4f}",
                r["deviceId"], r["streamId"],
                grid, block,
                r["registersPerThread"],
                r["sharedMem"],
                "kernel",
                short_name(r["demangledName"]),
                r["demangledName"] or "",
            ])
    print(f"  {len(kernels)} kernels → {kernel_csv}")

    # ── Memcpy (GPU-side) ────────────────────────────────────────────────
    memcpys = conn.execute(MEMCPY_SQL.format(lo=lo, hi=hi)).fetchall()
    memcpy_csv = os.path.join(args.out_dir, f"memcpy_{tag}.csv")
    with open(memcpy_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "start_ms", "end_ms", "duration_ms",
            "size_bytes", "size_human",
            "src_mem", "dst_mem",
            "device_id", "stream_id",
            "dma_bw_gbps",
            "op_type", "direction",
        ])
        for r in memcpys:
            dur_ns = r["duration"]
            bw = r["bytes"] / (dur_ns / 1e9) / 1e9 if dur_ns > 0 and r["bytes"] > 0 else 0
            w.writerow([
                f"{(r['start'] - t0) / 1e6:.4f}",
                f"{(r['end'] - t0) / 1e6:.4f}",
                f"{dur_ns / 1e6:.4f}",
                r["bytes"],
                fmt_bytes(r["bytes"]),
                MEM_KIND.get(r["srcKind"], str(r["srcKind"])),
                MEM_KIND.get(r["dstKind"], str(r["dstKind"])),
                r["deviceId"], r["streamId"],
                f"{bw:.2f}",
                "memcpy",
                COPY_KIND.get(r["copyKind"], str(r["copyKind"])),
            ])
    print(f"  {len(memcpys)} memcpy → {memcpy_csv}")

    # ── Memcpy E2E (CPU + GPU joined) ───────────────────────────────────
    e2e_rows = conn.execute(MEMCPY_E2E_SQL.format(lo=lo, hi=hi)).fetchall()
    e2e_csv = os.path.join(args.out_dir, f"memcpy_e2e_{tag}.csv")
    with open(e2e_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "cpu_start_ms", "cpu_end_ms", "cpu_wall_ms",
            "gpu_start_ms", "gpu_end_ms", "gpu_dma_ms",
            "launch_overhead_ms", "e2e_ms",
            "size_bytes", "size_human",
            "src_mem", "dst_mem",
            "device_id", "stream_id",
            "dma_bw_gbps",
            "op_type", "direction", "api_name",
        ])
        for r in e2e_rows:
            cpu_wall = (r["cpu_end"] - r["cpu_start"])
            gpu_dma = (r["gpu_end"] - r["gpu_start"])
            launch_oh = max(0, r["gpu_start"] - r["cpu_start"])
            e2e = max(cpu_wall, r["gpu_end"] - r["cpu_start"])
            bw = r["bytes"] / (gpu_dma / 1e9) / 1e9 if gpu_dma > 0 and r["bytes"] > 0 else 0
            w.writerow([
                f"{(r['cpu_start'] - t0) / 1e6:.4f}",
                f"{(r['cpu_end'] - t0) / 1e6:.4f}",
                f"{cpu_wall / 1e6:.4f}",
                f"{(r['gpu_start'] - t0) / 1e6:.4f}",
                f"{(r['gpu_end'] - t0) / 1e6:.4f}",
                f"{gpu_dma / 1e6:.4f}",
                f"{launch_oh / 1e6:.4f}",
                f"{e2e / 1e6:.4f}",
                r["bytes"],
                fmt_bytes(r["bytes"]),
                MEM_KIND.get(r["srcKind"], str(r["srcKind"])),
                MEM_KIND.get(r["dstKind"], str(r["dstKind"])),
                r["deviceId"], r["streamId"],
                f"{bw:.2f}",
                "memcpy",
                COPY_KIND.get(r["copyKind"], str(r["copyKind"])),
                r["api_name"] or "",
            ])
    print(f"  {len(e2e_rows)} memcpy e2e → {e2e_csv}")

    conn.close()

    print(f"\nDumped {args.t1}–{args.t2} ms window to {args.out_dir}/")


if __name__ == "__main__":
    main()
