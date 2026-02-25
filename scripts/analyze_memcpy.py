#!/usr/bin/env python3
"""Analyze true end-to-end memcpy timing from Nsight Systems traces.

Joins CPU-side CUDA runtime API calls with GPU-side DMA activity
via correlationId to measure the FULL transfer pipeline:

    CPU call start ──► GPU DMA start ──► GPU DMA end ──► CPU call return
    │← launch overhead →│←── GPU DMA ──→│← sync wait →│
    │←──────────── CPU wall clock ─────────────────────→│

Usage:
    python scripts/analyze_memcpy.py --sqlite nsys_reports/*.sqlite
    python scripts/analyze_memcpy.py --sqlite nsys_reports/*.sqlite --start-ms 1000 --end-ms 3300
    python scripts/analyze_memcpy.py --sqlite nsys_reports/*.sqlite --csv memcpy_e2e.csv
"""
import argparse
import csv
import sqlite3
import sys
from collections import defaultdict

COPY_KIND = {0: "Unknown", 1: "HtoD", 2: "DtoH", 3: "HtoH", 4: "DtoD", 8: "Peer"}
MEM_KIND = {0: "Unknown", 1: "Pageable", 2: "Device", 3: "Array", 4: "Unified", 5: "Managed"}

E2E_QUERY = """
SELECT
    r.start        AS cpu_start,
    r.end          AS cpu_end,
    r.end - r.start AS cpu_wall_ns,
    m.start        AS gpu_start,
    m.end          AS gpu_end,
    m.end - m.start AS gpu_dma_ns,
    m.start - r.start AS launch_overhead_ns,
    r.end   - m.end   AS sync_wait_ns,
    m.bytes,
    m.copyKind,
    m.srcKind,
    m.dstKind,
    m.deviceId,
    m.streamId,
    m.correlationId,
    s.value        AS api_name
FROM CUPTI_ACTIVITY_KIND_MEMCPY AS m
JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
    ON m.correlationId = r.correlationId
LEFT JOIN StringIds AS s ON r.nameId = s.id
{where}
ORDER BY m.start
"""


def fmt_bytes(b):
    if b >= 1 << 30:
        return f"{b / (1 << 30):.2f} GiB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.1f} MiB"
    if b >= 1 << 10:
        return f"{b / (1 << 10):.1f} KiB"
    return f"{b} B"


def analyze(sqlite_path, start_ms=None, end_ms=None, csv_path=None):
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row

    t0 = conn.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    if t0 is None:
        t0 = conn.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchone()[0]

    # Build WHERE clause
    conditions = []
    if start_ms is not None:
        conditions.append(f"m.start >= {t0 + int(start_ms * 1e6)}")
    if end_ms is not None:
        conditions.append(f"m.start <= {t0 + int(end_ms * 1e6)}")
    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    rows = [dict(r) for r in conn.execute(E2E_QUERY.format(where=where)).fetchall()]
    conn.close()

    if not rows:
        print("No memcpy events found.")
        return

    print(f"\nAnalyzing {len(rows)} memcpy events ...\n")

    # ── Per-direction summary ────────────────────────────────────────────────
    stats = defaultdict(lambda: {
        "count": 0, "bytes": 0,
        "cpu_wall": 0, "gpu_dma": 0, "launch_oh": 0, "sync_wait": 0,
        "e2e_list": [], "bw_list": [],
    })

    for r in rows:
        kind = COPY_KIND.get(r["copyKind"], "?")
        s = stats[kind]
        s["count"] += 1
        s["bytes"] += r["bytes"]
        s["cpu_wall"] += r["cpu_wall_ns"]
        s["gpu_dma"] += r["gpu_dma_ns"]
        s["launch_oh"] += max(0, r["launch_overhead_ns"])
        s["sync_wait"] += max(0, r["sync_wait_ns"])
        # E2E = max(cpu_wall, gpu_end - cpu_start) to capture full span
        e2e = max(r["cpu_wall_ns"], r["gpu_end"] - r["cpu_start"])
        s["e2e_list"].append(e2e)
        if r["gpu_dma_ns"] > 0 and r["bytes"] > 0:
            s["bw_list"].append(r["bytes"] / (r["gpu_dma_ns"] / 1e9) / 1e9)

    print("=" * 105)
    print(f"{'Direction':<10s} {'Count':>7s} {'Total Size':>12s} │ {'CPU Wall':>10s} {'GPU DMA':>10s} {'Launch OH':>10s} {'Sync Wait':>10s} │ {'DMA BW':>10s}")
    print("=" * 105)
    for kind in ["HtoD", "DtoH", "DtoD", "Peer", "HtoH", "Unknown"]:
        s = stats.get(kind)
        if not s or s["count"] == 0:
            continue
        bw = sum(s["bw_list"]) / len(s["bw_list"]) if s["bw_list"] else 0
        print(
            f"{kind:<10s} {s['count']:>7d} {fmt_bytes(s['bytes']):>12s} │ "
            f"{s['cpu_wall']/1e6:>8.1f}ms {s['gpu_dma']/1e6:>8.1f}ms "
            f"{s['launch_oh']/1e6:>8.1f}ms {s['sync_wait']/1e6:>8.1f}ms │ "
            f"{bw:>8.1f} GB/s"
        )

    # ── Top transfers by E2E time ────────────────────────────────────────────
    rows_sorted = sorted(rows, key=lambda r: max(r["cpu_wall_ns"], r["gpu_end"] - r["cpu_start"]), reverse=True)

    print(f"\n{'─' * 105}")
    print(f"Top 15 memcpy by end-to-end time (CPU call start → GPU DMA end):\n")
    print(f"  {'#':>3s} {'Dir':<6s} {'Size':>10s} │ {'E2E':>10s} {'CPU Wall':>10s} {'GPU DMA':>10s} {'Launch OH':>10s} {'Sync':>10s} │ {'BW':>10s}")
    print(f"  {'─' * 95}")

    for i, r in enumerate(rows_sorted[:15]):
        kind = COPY_KIND.get(r["copyKind"], "?")
        e2e = max(r["cpu_wall_ns"], r["gpu_end"] - r["cpu_start"])
        bw = r["bytes"] / (r["gpu_dma_ns"] / 1e9) / 1e9 if r["gpu_dma_ns"] > 0 and r["bytes"] > 0 else 0
        launch = max(0, r["launch_overhead_ns"])
        sync = max(0, r["sync_wait_ns"])
        print(
            f"  {i+1:>3d} {kind:<6s} {fmt_bytes(r['bytes']):>10s} │ "
            f"{e2e/1e3:>8.1f}us {r['cpu_wall_ns']/1e3:>8.1f}us {r['gpu_dma_ns']/1e3:>8.1f}us "
            f"{launch/1e3:>8.1f}us {sync/1e3:>8.1f}us │ "
            f"{bw:>8.1f} GB/s"
        )

    # ── Pageable vs Pinned breakdown ─────────────────────────────────────────
    print(f"\n{'─' * 105}")
    print("Memory type breakdown (pageable vs pinned affects transfer speed):\n")
    mem_stats = defaultdict(lambda: {"count": 0, "bytes": 0, "gpu_dma": 0})
    for r in rows:
        sk = MEM_KIND.get(r["srcKind"], "?")
        dk = MEM_KIND.get(r["dstKind"], "?")
        key = f"{sk} → {dk}"
        m = mem_stats[key]
        m["count"] += 1
        m["bytes"] += r["bytes"]
        m["gpu_dma"] += r["gpu_dma_ns"]

    for key, m in sorted(mem_stats.items(), key=lambda x: x[1]["bytes"], reverse=True):
        bw = m["bytes"] / (m["gpu_dma"] / 1e9) / 1e9 if m["gpu_dma"] > 0 else 0
        print(f"  {key:<30s}  {m['count']:>6d} transfers  {fmt_bytes(m['bytes']):>10s}  {m['gpu_dma']/1e6:>8.1f}ms GPU DMA  {bw:>8.1f} GB/s")

    print()
    print("Legend:")
    print("  CPU Wall    = time from cudaMemcpyAsync() call to return (what your app sees)")
    print("  GPU DMA     = time the DMA engine is actually moving bytes on the bus")
    print("  Launch OH   = cpu_call_start → gpu_dma_start (pinning, staging, scheduling)")
    print("  Sync Wait   = gpu_dma_end → cpu_call_return (only nonzero for sync copies)")
    print("  E2E         = max(CPU Wall, gpu_dma_end - cpu_call_start)")
    print("  DMA BW      = bytes / GPU DMA time (effective PCIe bandwidth)")

    # ── Optional CSV export ──────────────────────────────────────────────────
    if csv_path:
        header = [
            "copy_kind", "src_mem", "dst_mem", "bytes",
            "cpu_start_ns", "cpu_end_ns", "cpu_wall_ns",
            "gpu_start_ns", "gpu_end_ns", "gpu_dma_ns",
            "launch_overhead_ns", "sync_wait_ns", "e2e_ns",
            "dma_bw_gbps", "api_name", "stream_id", "device_id",
        ]
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                kind = COPY_KIND.get(r["copyKind"], "?")
                sk = MEM_KIND.get(r["srcKind"], "?")
                dk = MEM_KIND.get(r["dstKind"], "?")
                e2e = max(r["cpu_wall_ns"], r["gpu_end"] - r["cpu_start"])
                bw = r["bytes"] / (r["gpu_dma_ns"] / 1e9) / 1e9 if r["gpu_dma_ns"] > 0 and r["bytes"] > 0 else 0
                w.writerow([
                    kind, sk, dk, r["bytes"],
                    r["cpu_start"], r["cpu_end"], r["cpu_wall_ns"],
                    r["gpu_start"], r["gpu_end"], r["gpu_dma_ns"],
                    max(0, r["launch_overhead_ns"]),
                    max(0, r["sync_wait_ns"]),
                    e2e, f"{bw:.2f}",
                    r["api_name"], r["streamId"], r["deviceId"],
                ])
        print(f"\nCSV exported: {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Analyze end-to-end memcpy timing")
    p.add_argument("--sqlite", required=True)
    p.add_argument("--start-ms", type=float, default=None)
    p.add_argument("--end-ms", type=float, default=None)
    p.add_argument("--csv", default=None, help="Export detailed CSV")
    args = p.parse_args()
    analyze(args.sqlite, args.start_ms, args.end_ms, args.csv)
