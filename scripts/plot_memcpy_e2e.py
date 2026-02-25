#!/usr/bin/env python3
"""Visualize end-to-end memcpy timing: CPU API call vs GPU DMA on the SAME plot.

Shows two rows per memcpy direction:
  - CPU row: when cudaMemcpyAsync() was called and returned
  - GPU row: when the DMA engine actually moved bytes

The gap between them is the launch overhead (pinning, staging, scheduling).

Usage:
    python scripts/plot_memcpy_e2e.py --sqlite nsys_reports/*.sqlite
    python scripts/plot_memcpy_e2e.py --sqlite nsys_reports/*.sqlite --end-ms 1000  # model loading only
    python scripts/plot_memcpy_e2e.py --sqlite nsys_reports/*.sqlite --start-ms 1000 --end-ms 1050
    python scripts/plot_memcpy_e2e.py --sqlite nsys_reports/*.sqlite --min-bytes 1048576  # only transfers >= 1MiB
"""
import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

COPY_KIND = {0: "Unknown", 1: "HtoD", 2: "DtoH", 3: "HtoH", 4: "DtoD", 8: "Peer"}
MEM_KIND = {0: "Unknown", 1: "Pageable", 2: "Device", 3: "Array", 4: "Unified", 5: "Managed"}

E2E_QUERY = """
SELECT
    r.start        AS cpu_start,
    r.end          AS cpu_end,
    m.start        AS gpu_start,
    m.end          AS gpu_end,
    m.bytes,
    m.copyKind,
    m.srcKind,
    m.dstKind,
    m.streamId,
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
        return f"{b / (1 << 30):.1f} GiB"
    if b >= 1 << 20:
        return f"{b / (1 << 20):.1f} MiB"
    if b >= 1 << 10:
        return f"{b / (1 << 10):.1f} KiB"
    return f"{b} B"


def load(sqlite_path, start_ms=None, end_ms=None, min_bytes=0):
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row

    t0 = conn.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    if t0 is None:
        t0 = conn.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchone()[0]

    conditions = []
    if start_ms is not None:
        conditions.append(f"m.start >= {t0 + int(start_ms * 1e6)}")
    if end_ms is not None:
        conditions.append(f"m.start <= {t0 + int(end_ms * 1e6)}")
    if min_bytes > 0:
        conditions.append(f"m.bytes >= {min_bytes}")
    where = "WHERE " + " AND ".join(conditions) if conditions else ""

    rows = [dict(r) for r in conn.execute(E2E_QUERY.format(where=where)).fetchall()]
    conn.close()

    if not rows:
        print("No memcpy events found.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    df["kind"] = df["copyKind"].map(COPY_KIND)
    df["src_mem"] = df["srcKind"].map(MEM_KIND)
    df["dst_mem"] = df["dstKind"].map(MEM_KIND)

    # Convert to ms relative to t0
    for col in ["cpu_start", "cpu_end", "gpu_start", "gpu_end"]:
        df[col + "_ms"] = (df[col] - t0) / 1e6

    df["cpu_dur_ms"] = df["cpu_end_ms"] - df["cpu_start_ms"]
    df["gpu_dur_ms"] = df["gpu_end_ms"] - df["gpu_start_ms"]
    df["launch_oh_ms"] = (df["gpu_start_ms"] - df["cpu_start_ms"]).clip(lower=0)
    df["e2e_ms"] = df[["cpu_end_ms", "gpu_end_ms"]].max(axis=1) - df["cpu_start_ms"]

    print(f"  Loaded {len(df)} memcpy events")
    return df, t0


def plot(df, output_file, title="Memcpy E2E: CPU API call vs GPU DMA"):
    # Colors
    kind_colors = {
        "HtoD": {"cpu": "#ef9a9a", "gpu": "#c62828"},  # light red / dark red
        "DtoH": {"cpu": "#a5d6a7", "gpu": "#2e7d32"},  # light green / dark green
        "DtoD": {"cpu": "#ffe0b2", "gpu": "#e65100"},  # light orange / dark orange
        "Peer": {"cpu": "#ce93d8", "gpu": "#6a1b9a"},  # light purple / dark purple
        "HtoH": {"cpu": "#90caf9", "gpu": "#1565c0"},  # light blue / dark blue
    }
    default_colors = {"cpu": "#bdbdbd", "gpu": "#424242"}

    # Minimum visible bar width
    view_span = df["gpu_end_ms"].max() - df["cpu_start_ms"].min()
    min_bar = max(view_span * 0.001, 0.001)

    fig = go.Figure()

    for kind in ["HtoD", "DtoH", "DtoD", "Peer"]:
        sub = df[df["kind"] == kind]
        if sub.empty:
            continue

        colors = kind_colors.get(kind, default_colors)

        # CPU bar (light color) — the API call duration
        cpu_dur_vis = sub["cpu_dur_ms"].clip(lower=min_bar)
        fig.add_trace(go.Bar(
            name=f"{kind} — CPU call",
            y=[f"Stream {s} — CPU" for s in sub["streamId"]],
            x=cpu_dur_vis,
            base=sub["cpu_start_ms"],
            orientation="h",
            marker=dict(color=colors["cpu"], line=dict(color=colors["cpu"], width=1)),
            hoverinfo="text",
            hovertext=sub.apply(lambda r: (
                f"<b>CPU: {r['api_name'] or 'cudaMemcpy'}</b><br>"
                f"Direction: {r['kind']} ({r['src_mem']} → {r['dst_mem']})<br>"
                f"Size: {fmt_bytes(int(r['bytes']))}<br>"
                f"CPU call: {r['cpu_start_ms']:.4f} → {r['cpu_end_ms']:.4f} ms "
                f"({r['cpu_dur_ms']:.4f} ms)<br>"
                f"Launch overhead: {r['launch_oh_ms']:.4f} ms<br>"
                f"E2E: {r['e2e_ms']:.4f} ms"
            ), axis=1),
            legendgroup=kind,
            showlegend=True,
        ))

        # GPU bar (dark color) — the actual DMA transfer
        gpu_dur_vis = sub["gpu_dur_ms"].clip(lower=min_bar)
        fig.add_trace(go.Bar(
            name=f"{kind} — GPU DMA",
            y=[f"Stream {s} — GPU" for s in sub["streamId"]],
            x=gpu_dur_vis,
            base=sub["gpu_start_ms"],
            orientation="h",
            marker=dict(color=colors["gpu"], line=dict(color=colors["gpu"], width=1)),
            hoverinfo="text",
            hovertext=sub.apply(lambda r: (
                f"<b>GPU DMA: {r['kind']}</b><br>"
                f"Size: {fmt_bytes(int(r['bytes']))}<br>"
                f"GPU DMA: {r['gpu_start_ms']:.4f} → {r['gpu_end_ms']:.4f} ms "
                f"({r['gpu_dur_ms']:.4f} ms)<br>"
                f"BW: {r['bytes'] / (r['gpu_dur_ms'] / 1e3) / 1e9:.1f} GB/s"
                if r['gpu_dur_ms'] > 0 else
                f"<b>GPU DMA: {r['kind']}</b><br>Size: {fmt_bytes(int(r['bytes']))}<br>~instant"
            ), axis=1),
            legendgroup=kind,
            showlegend=True,
        ))

    # Compute unique tracks and order them: CPU above GPU for each stream
    streams = sorted(df["streamId"].unique())
    track_order = []
    for s in streams:
        track_order.append(f"Stream {s} — CPU")
        track_order.append(f"Stream {s} — GPU")

    height = max(400, 100 + 60 * len(track_order))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Time (ms)",
        yaxis_title="",
        height=height,
        margin=dict(l=180, r=50, t=80, b=50),
        plot_bgcolor="white",
        hovermode="closest",
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(size=11),
        ),
        barmode="overlay",
        bargap=0.3,
    )
    fig.update_yaxes(categoryorder="array", categoryarray=track_order)
    fig.update_xaxes(showgrid=True, gridcolor="#eee")

    # Add annotation explaining the layout
    fig.add_annotation(
        text="Light bars = CPU API call duration  |  Dark bars = GPU DMA engine active  |  Gap = launch overhead",
        xref="paper", yref="paper", x=0.5, y=-0.06,
        showarrow=False, font=dict(size=11, color="#666"),
    )

    fig.write_html(output_file)
    print(f"Plot saved to {output_file}")


def main():
    p = argparse.ArgumentParser(description="Visualize CPU↔GPU memcpy correlation")
    p.add_argument("--sqlite", required=True)
    p.add_argument("--start-ms", type=float, default=None)
    p.add_argument("--end-ms", type=float, default=None)
    p.add_argument("--min-bytes", type=int, default=0, help="Only show transfers >= N bytes")
    p.add_argument("--output", default=None)
    p.add_argument("--title", default="Memcpy E2E: CPU API call vs GPU DMA")
    args = p.parse_args()

    print(f"Loading from {args.sqlite} ...")
    df, t0 = load(args.sqlite, args.start_ms, args.end_ms, args.min_bytes)

    if args.output:
        out = args.output
    else:
        stem = Path(args.sqlite).stem
        suffix = ""
        if args.start_ms is not None or args.end_ms is not None:
            suffix = f"_{int(args.start_ms or 0)}-{int(args.end_ms or 99999)}ms"
        if args.min_bytes > 0:
            suffix += f"_min{fmt_bytes(args.min_bytes).replace(' ', '')}"
        out = str(Path(args.sqlite).parent / f"{stem}_memcpy_e2e{suffix}.html")

    plot(df, out, title=args.title)


if __name__ == "__main__":
    main()
