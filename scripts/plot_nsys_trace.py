#!/usr/bin/env python3
"""Plotly-based Nsight Systems trace visualization.

Adapted from the engine_tasks performance trace plotter.
Reads the nsys SQLite database directly and produces an interactive HTML
horizontal bar chart showing kernel and memcpy events on per-stream tracks.

Usage:
    python scripts/plot_nsys_trace.py [options]

Examples:
    # Single inference run, 50ms window
    python scripts/plot_nsys_trace.py --sqlite nsys_reports/*.sqlite --start-ms 1000 --end-ms 1050

    # Wider view with aggregation (bins kernels into 0.1ms buckets)
    python scripts/plot_nsys_trace.py --sqlite nsys_reports/*.sqlite --start-ms 1000 --end-ms 3300 --bin-ms 0.1

    # Top 200 longest kernels across the whole trace
    python scripts/plot_nsys_trace.py --sqlite nsys_reports/*.sqlite --top-kernels 200
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ── Lookup tables ────────────────────────────────────────────────────────────
MEMCPY_KIND = {0: "Unknown", 1: "HtoD", 2: "DtoH", 3: "HtoH", 4: "DtoD", 8: "Peer"}
MEM_KIND = {0: "Unknown", 1: "Pageable", 2: "Device", 3: "Array", 4: "Unified", 5: "Managed"}


def fmt_bytes(b: int) -> str:
    if b >= 1 << 20:
        return f"{b / (1 << 20):.1f} MiB"
    if b >= 1 << 10:
        return f"{b / (1 << 10):.1f} KiB"
    return f"{b} B"


def short_name(demangled: str) -> str:
    """Readable short name from a demangled CUDA kernel name."""
    if not demangled:
        return "unknown"
    name = demangled.split("<")[0].split("(")[0]
    parts = name.split("::")
    return "::".join(parts[-2:]) if len(parts) > 1 else parts[-1]


# ── Data loading ─────────────────────────────────────────────────────────────

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
{where}
ORDER BY k.start
"""

MEMCPY_SQL = """
SELECT
    start, end, end - start AS duration,
    bytes, copyKind, srcKind, dstKind,
    deviceId, streamId
FROM CUPTI_ACTIVITY_KIND_MEMCPY
{where}
ORDER BY start
"""


def load_events(sqlite_path: str, start_ms=None, end_ms=None, top_kernels=None, bin_ms=None):
    """Load kernel + memcpy events from the nsys sqlite database."""
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row

    # Global time base
    t0 = conn.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    if t0 is None:
        t0 = conn.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchone()[0]
    if t0 is None:
        print("No events found in database.")
        sys.exit(1)

    # Build WHERE clauses for time window
    k_where, m_where = "", ""
    if start_ms is not None or end_ms is not None:
        conditions = []
        if start_ms is not None:
            lo = t0 + int(start_ms * 1e6)
            conditions.append(f"k.end >= {lo}")
        if end_ms is not None:
            hi = t0 + int(end_ms * 1e6)
            conditions.append(f"k.start <= {hi}")
        k_where = "WHERE " + " AND ".join(conditions)
        m_where = k_where.replace("k.", "")

    # ── Kernels ──
    kernel_rows = conn.execute(KERNEL_SQL.format(where=k_where)).fetchall()
    print(f"  Loaded {len(kernel_rows)} kernels")

    records = []
    for r in kernel_rows:
        sname = short_name(r["demangledName"] or "")
        grid = f"{r['gridX']}x{r['gridY']}x{r['gridZ']}"
        block = f"{r['blockX']}x{r['blockY']}x{r['blockZ']}"
        records.append({
            "event_type": "kernel",
            "op_name": sname,
            "full_name": r["demangledName"] or "",
            "start_ms": (r["start"] - t0) / 1e6,
            "end_ms": (r["end"] - t0) / 1e6,
            "duration_ms": r["duration"] / 1e6,
            "stream_id": r["streamId"],
            "device_id": r["deviceId"],
            "bytes": 0,
            "detail": f"grid={grid} block={block} regs={r['registersPerThread']} shmem={fmt_bytes(r['sharedMem'])}",
        })

    # ── Memcpy ──
    memcpy_rows = conn.execute(MEMCPY_SQL.format(where=m_where)).fetchall()
    print(f"  Loaded {len(memcpy_rows)} memcpys")
    conn.close()

    for r in memcpy_rows:
        ck = MEMCPY_KIND.get(r["copyKind"], str(r["copyKind"]))
        sk = MEM_KIND.get(r["srcKind"], str(r["srcKind"]))
        dk = MEM_KIND.get(r["dstKind"], str(r["dstKind"]))
        records.append({
            "event_type": "memcpy",
            "op_name": f"{ck} ({fmt_bytes(r['bytes'])})",
            "full_name": ck,
            "start_ms": (r["start"] - t0) / 1e6,
            "end_ms": (r["end"] - t0) / 1e6,
            "duration_ms": r["duration"] / 1e6,
            "stream_id": r["streamId"],
            "device_id": r["deviceId"],
            "bytes": r["bytes"],
            "detail": f"{sk} → {dk}",
        })

    df = pd.DataFrame(records)
    if df.empty:
        print("No events in the selected window.")
        sys.exit(1)

    # ── Top-N filter ──
    if top_kernels is not None:
        kernels = df[df["event_type"] == "kernel"].nlargest(top_kernels, "duration_ms")
        memcpys = df[df["event_type"] == "memcpy"]
        df = pd.concat([kernels, memcpys]).sort_values("start_ms").reset_index(drop=True)

    # ── Binning / aggregation ──
    if bin_ms is not None and bin_ms > 0:
        df = bin_events(df, bin_ms)

    print(f"  Final: {len(df)} events for plotting")
    return df, t0


def bin_events(df: pd.DataFrame, bin_ms: float) -> pd.DataFrame:
    """Aggregate kernels into fixed-width time bins to reduce event count.

    Memcpy events are kept as-is (they're already sparse).
    """
    memcpys = df[df["event_type"] == "memcpy"].copy()
    kernels = df[df["event_type"] == "kernel"].copy()

    if kernels.empty:
        return memcpys

    kernels["bin"] = (kernels["start_ms"] / bin_ms).astype(int)

    agg = kernels.groupby(["stream_id", "bin"]).agg(
        count=("op_name", "size"),
        start_ms=("start_ms", "min"),
        end_ms=("end_ms", "max"),
        total_dur=("duration_ms", "sum"),
        top_kernel=("op_name", lambda s: s.value_counts().index[0]),
        device_id=("device_id", "first"),
    ).reset_index()

    agg["duration_ms"] = agg["end_ms"] - agg["start_ms"]
    agg["event_type"] = "kernel_bin"
    agg["op_name"] = agg.apply(
        lambda r: f"{r['top_kernel']} (+{r['count']-1})" if r["count"] > 1 else r["top_kernel"],
        axis=1,
    )
    agg["full_name"] = agg["op_name"]
    agg["bytes"] = 0
    agg["detail"] = agg.apply(
        lambda r: f"{r['count']} kernels, {r['total_dur']:.3f} ms active", axis=1
    )

    cols = ["event_type", "op_name", "full_name", "start_ms", "end_ms",
            "duration_ms", "stream_id", "device_id", "bytes", "detail"]
    result = pd.concat([agg[cols], memcpys[cols]]).sort_values("start_ms").reset_index(drop=True)
    print(f"  Binned {len(kernels)} kernels → {len(agg)} bins ({bin_ms} ms each)")
    return result


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_trace(df: pd.DataFrame, output_file: str, title: str = "Nsight Systems Trace"):
    """Produce an interactive Plotly horizontal bar chart.

    Compute kernels and memcpy events share the SAME y-axis track (per stream),
    distinguished by color and legend entry.
    """
    df = df.copy()

    # One track per stream — kernels and memcpy on the same row
    df["track_name"] = df["stream_id"].apply(lambda s: f"Stream {s}")
    df["duration_ms"] = df["end_ms"] - df["start_ms"]
    df = df.sort_values(by=["track_name", "start_ms"]).reset_index(drop=True)

    # Assign a legend category to each event
    def legend_category(row):
        if row["event_type"] in ("kernel", "kernel_bin"):
            return "Compute (kernel)"
        # memcpy — subcategorize by direction
        fn = str(row.get("full_name", ""))
        if "HtoD" in fn:
            return "Memcpy HtoD"
        if "DtoH" in fn:
            return "Memcpy DtoH"
        if "DtoD" in fn:
            return "Memcpy DtoD"
        return "Memcpy (other)"

    df["category"] = df.apply(legend_category, axis=1)

    # Fixed color per legend category
    category_colors = {
        "Compute (kernel)": "#4fc3f7",
        "Memcpy HtoD":     "#ef5350",
        "Memcpy DtoH":     "#66bb6a",
        "Memcpy DtoD":     "#ffa726",
        "Memcpy (other)":  "#ab47bc",
    }

    # Hover text
    df["hover"] = df.apply(
        lambda r: (
            f"<b>{r['op_name']}</b><br>"
            f"Stream: {r['stream_id']}  |  {r['category']}<br>"
            f"Start: {r['start_ms']:.4f} ms<br>"
            f"End: {r['end_ms']:.4f} ms<br>"
            f"Duration: {r['duration_ms']:.4f} ms<br>"
            f"{r['detail']}"
            + (f"<br>Bytes: {fmt_bytes(int(r['bytes']))}" if r["bytes"] > 0 else "")
        ),
        axis=1,
    )

    unique_tracks = sorted(df["track_name"].unique())

    fig = go.Figure()

    # One trace per legend category so they share y-rows but get distinct legend entries
    for cat in ["Compute (kernel)", "Memcpy HtoD", "Memcpy DtoH", "Memcpy DtoD", "Memcpy (other)"]:
        sub = df[df["category"] == cat]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            name=cat,
            y=sub["track_name"],
            x=sub["duration_ms"],
            base=sub["start_ms"],
            orientation="h",
            marker=dict(
                color=category_colors[cat],
                line=dict(color="white", width=0.5),
            ),
            text=sub["op_name"],
            textposition="inside",
            insidetextanchor="middle",
            textfont=dict(size=9),
            hoverinfo="text",
            hovertext=sub["hover"],
            legendgroup=cat,
            showlegend=True,
        ))

    height = max(400, 150 + 80 * len(unique_tracks))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Time (ms)",
        yaxis_title="Stream",
        height=height,
        margin=dict(l=150, r=50, t=60, b=50),
        plot_bgcolor="white",
        hovermode="closest",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=12),
        ),
        barmode="overlay",  # kernels and memcpy overlap on the same row
        bargap=0.3,
        uniformtext_minsize=7,
        uniformtext_mode="hide",
    )
    fig.update_yaxes(categoryorder="array", categoryarray=unique_tracks)
    fig.update_xaxes(showgrid=True, gridcolor="#eee", title_standoff=10)

    fig.write_html(output_file)
    print(f"Plot saved to {output_file}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plotly nsys trace visualizer")
    parser.add_argument("--sqlite", required=True, help="Path to .sqlite file")
    parser.add_argument("--start-ms", type=float, default=None, help="Window start (ms from trace start)")
    parser.add_argument("--end-ms", type=float, default=None, help="Window end (ms)")
    parser.add_argument("--top-kernels", type=int, default=None, help="Only show N longest kernels")
    parser.add_argument("--bin-ms", type=float, default=None,
                        help="Aggregate kernels into time bins of this width (ms). "
                             "Use for wide time ranges to keep the plot interactive.")
    parser.add_argument("--output", default=None, help="Output HTML path (auto-generated if omitted)")
    parser.add_argument("--title", default="Nsight Systems Trace — Qwen3-VL-2B", help="Plot title")
    args = parser.parse_args()

    print(f"Loading events from {args.sqlite} ...")
    df, t0 = load_events(args.sqlite, args.start_ms, args.end_ms, args.top_kernels, args.bin_ms)

    if args.output:
        out = args.output
    else:
        stem = Path(args.sqlite).stem
        suffix = ""
        if args.start_ms is not None or args.end_ms is not None:
            suffix = f"_{int(args.start_ms or 0)}-{int(args.end_ms or 99999)}ms"
        if args.top_kernels:
            suffix += f"_top{args.top_kernels}"
        if args.bin_ms:
            suffix += f"_bin{args.bin_ms}ms"
        out = str(Path(args.sqlite).parent / f"{stem}{suffix}_plotly.html")

    plot_trace(df, out, title=args.title)


if __name__ == "__main__":
    main()
