#!/usr/bin/env python3
"""Export memcpy events from an Nsight Systems SQLite database to CSV.

Usage:
    python export_nsys_memcpy_csv.py <input.sqlite> [output.csv]
"""
import csv
import sqlite3
import sys


MEMCPY_KIND = {0: "Unknown", 1: "HtoD", 2: "DtoH", 3: "HtoH", 4: "DtoD", 8: "Peer"}
MEM_KIND = {0: "Unknown", 1: "Pageable", 2: "Device", 3: "Array", 4: "Unified", 5: "Managed"}

QUERY = """
SELECT
    start        AS start_ns,
    end          AS end_ns,
    end - start  AS duration_ns,
    bytes,
    copyKind,
    srcKind,
    dstKind,
    deviceId,
    contextId,
    streamId,
    correlationId,
    globalPid
FROM CUPTI_ACTIVITY_KIND_MEMCPY
ORDER BY start
"""


def export(sqlite_path: str, csv_path: str) -> int:
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(QUERY).fetchall()
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            print(f"  No CUPTI_ACTIVITY_KIND_MEMCPY table (0 memcpy events)")
            rows = []
        else:
            raise
    finally:
        conn.close()

    header = [
        "start_ns", "end_ns", "duration_ns", "bytes",
        "copy_kind", "src_mem_kind", "dst_mem_kind",
        "device_id", "context_id", "stream_id",
        "correlation_id", "global_pid",
    ]

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([
                r["start_ns"], r["end_ns"], r["duration_ns"], r["bytes"],
                MEMCPY_KIND.get(r["copyKind"], r["copyKind"]),
                MEM_KIND.get(r["srcKind"], r["srcKind"]),
                MEM_KIND.get(r["dstKind"], r["dstKind"]),
                r["deviceId"], r["contextId"], r["streamId"],
                r["correlationId"], r["globalPid"],
            ])

    print(f"  {len(rows)} memcpy events → {csv_path}")
    return len(rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sqlite_path = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else sqlite_path.replace(".sqlite", "_memcpy_trace.csv")
    export(sqlite_path, csv_path)
