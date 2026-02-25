#!/usr/bin/env python3
"""Dump kernel-to-kernel memory copy pairs from an nsys trace.

For each memcpy, shows which kernel ran before it and which kernel runs after —
revealing the data flow between compute operations.

Usage:
    python scripts/dump_kernel_memcpy_pairs.py --sqlite nsys_reports/*.sqlite --t1 2260 --t2 2400
    python scripts/dump_kernel_memcpy_pairs.py --sqlite nsys_reports/*.sqlite --t1 2260 --t2 2400 --out-dir selected_csv
"""
import argparse
import csv
import os
import sqlite3
import sys


COPY_KIND = {0: "Unknown", 1: "HtoD", 2: "DtoH", 3: "HtoH", 4: "DtoD", 8: "Peer"}
MEM_KIND = {0: "Unknown", 1: "Pageable", 2: "Device", 3: "Array", 4: "Unified", 5: "Managed"}


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
    p = argparse.ArgumentParser(description="Dump kernel↔memcpy pairs")
    p.add_argument("--sqlite", required=True)
    p.add_argument("--t1", type=float, required=True, help="Window start (ms)")
    p.add_argument("--t2", type=float, required=True, help="Window end (ms)")
    p.add_argument("--out-dir", default="selected_csv")
    args = p.parse_args()

    conn = sqlite3.connect(args.sqlite)
    conn.row_factory = sqlite3.Row

    t0 = conn.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    lo = t0 + int(args.t1 * 1e6)
    hi = t0 + int(args.t2 * 1e6)

    # Load all kernels and memcpy in the window, sorted by start time
    kernels = conn.execute("""
        SELECT k.start, k.end, k.streamId, s.value AS name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds s ON k.demangledName = s.id
        WHERE k.end >= ? AND k.start <= ?
        ORDER BY k.start
    """, (lo, hi)).fetchall()

    memcpys = conn.execute("""
        SELECT start, end, bytes, copyKind, srcKind, dstKind, streamId
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE end >= ? AND start <= ?
        ORDER BY start
    """, (lo, hi)).fetchall()

    conn.close()

    if not memcpys:
        print("No memcpy events in window.")
        return

    # Build merged timeline
    events = []
    for k in kernels:
        events.append({
            "type": "kernel",
            "start": k["start"],
            "end": k["end"],
            "stream": k["streamId"],
            "name": short_name(k["name"]),
            "full_name": k["name"] or "",
        })
    for m in memcpys:
        events.append({
            "type": "memcpy",
            "start": m["start"],
            "end": m["end"],
            "stream": m["streamId"],
            "bytes": m["bytes"],
            "copyKind": m["copyKind"],
            "srcKind": m["srcKind"],
            "dstKind": m["dstKind"],
        })
    events.sort(key=lambda e: e["start"])

    # For each memcpy, find the preceding and following kernel on the same stream
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{int(args.t1)}-{int(args.t2)}ms"
    csv_path = os.path.join(args.out_dir, f"kernel_memcpy_pairs_{tag}.csv")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "memcpy_start_ms", "memcpy_end_ms", "memcpy_dur_ms",
            "size_bytes", "size_human", "direction", "src_mem", "dst_mem",
            "gap_before_ms", "gap_after_ms",
            "stream_id", "dma_bw_gbps",
            "prev_kernel_name", "prev_kernel_end_ms",
            "next_kernel_name", "next_kernel_start_ms",
        ])

        for i, ev in enumerate(events):
            if ev["type"] != "memcpy":
                continue

            stream = ev["stream"]

            # Search backward for preceding kernel on same stream
            prev_kernel = None
            for j in range(i - 1, -1, -1):
                if events[j]["type"] == "kernel" and events[j]["stream"] == stream:
                    prev_kernel = events[j]
                    break

            # Search forward for following kernel on same stream
            next_kernel = None
            for j in range(i + 1, len(events)):
                if events[j]["type"] == "kernel" and events[j]["stream"] == stream:
                    next_kernel = events[j]
                    break

            dur_ns = ev["end"] - ev["start"]
            bw = ev["bytes"] / (dur_ns / 1e9) / 1e9 if dur_ns > 0 and ev["bytes"] > 0 else 0
            gap_before = (ev["start"] - prev_kernel["end"]) / 1e6 if prev_kernel else None
            gap_after = (next_kernel["start"] - ev["end"]) / 1e6 if next_kernel else None

            w.writerow([
                f"{(ev['start'] - t0) / 1e6:.4f}",
                f"{(ev['end'] - t0) / 1e6:.4f}",
                f"{dur_ns / 1e6:.4f}",
                ev["bytes"],
                fmt_bytes(ev["bytes"]),
                COPY_KIND.get(ev["copyKind"], "?"),
                MEM_KIND.get(ev["srcKind"], "?"),
                MEM_KIND.get(ev["dstKind"], "?"),
                f"{gap_before:.4f}" if gap_before is not None else "",
                f"{gap_after:.4f}" if gap_after is not None else "",
                stream,
                f"{bw:.2f}",
                prev_kernel["name"] if prev_kernel else "",
                f"{(prev_kernel['end'] - t0) / 1e6:.4f}" if prev_kernel else "",
                next_kernel["name"] if next_kernel else "",
                f"{(next_kernel['start'] - t0) / 1e6:.4f}" if next_kernel else "",
            ])

    print(f"  {len(memcpys)} kernel↔memcpy pairs → {csv_path}")

    # Also print a summary to terminal
    print(f"\n  Timeline ({args.t1}–{args.t2} ms): {len(kernels)} kernels, {len(memcpys)} memcpy\n")
    print(f"  {'memcpy_ms':>10s} {'dur(us)':>8s} {'size':>10s} {'dir':<5s} {'gap_before':>10s} {'gap_after':>10s}  prev_kernel → next_kernel")
    print(f"  {'-' * 100}")

    # Re-read CSV for display
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gb = row["gap_before_ms"]
            ga = row["gap_after_ms"]
            print(
                f"  {float(row['memcpy_start_ms']):10.3f} "
                f"{float(row['memcpy_dur_ms'])*1000:8.1f} "
                f"{row['size_human']:>10s} "
                f"{row['direction']:<5s} "
                f"{float(gb)*1000 if gb else 0:8.1f}us "
                f"{float(ga)*1000 if ga else 0:8.1f}us  "
                f"{row['prev_kernel_name'][:30]:<30s} → {row['next_kernel_name'][:30]}"
            )


if __name__ == "__main__":
    main()
