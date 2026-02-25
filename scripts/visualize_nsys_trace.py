#!/usr/bin/env python3
"""Visualize Nsight Systems trace data as CSV + interactive HTML timeline.

Accepts .nsys-rep (auto-exports to .sqlite) or .sqlite directly.

Usage:
    python visualize_nsys_trace.py --input <file.nsys-rep|file.sqlite> [options]

Options:
    --start-ms FLOAT   Crop window start (ms from trace start)
    --end-ms FLOAT     Crop window end (ms from trace start)
    --top-kernels N    Only show the N longest-running kernels
    --nsys-bin PATH    Path to nsys binary
"""
import argparse
import csv
import html
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


DEFAULT_NSYS = "/home/a1/nsight-systems/opt/nvidia/nsight-systems/2026.1.1/bin/nsys"

# ── SQL queries ──────────────────────────────────────────────────────────────

KERNEL_QUERY = """
SELECT
    k.start,
    k.end,
    k.end - k.start AS duration,
    k.deviceId,
    k.streamId,
    k.gridX, k.gridY, k.gridZ,
    k.blockX, k.blockY, k.blockZ,
    k.registersPerThread,
    k.staticSharedMemory + k.dynamicSharedMemory AS sharedMem,
    s.value AS demangledName
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
LEFT JOIN StringIds AS s ON k.demangledName = s.id
ORDER BY k.start
"""

MEMCPY_QUERY = """
SELECT
    start,
    end,
    end - start AS duration,
    bytes,
    copyKind,
    srcKind,
    dstKind,
    deviceId,
    streamId
FROM CUPTI_ACTIVITY_KIND_MEMCPY
ORDER BY start
"""

MEMCPY_KIND = {0: "Unknown", 1: "HtoD", 2: "DtoH", 3: "HtoH", 4: "DtoD", 8: "Peer"}
MEM_KIND = {0: "Unknown", 1: "Pageable", 2: "Device", 3: "Array", 4: "Unified", 5: "Managed"}


def ensure_sqlite(input_path: str, nsys_bin: str) -> str:
    """Convert .nsys-rep → .sqlite if needed, return sqlite path."""
    p = Path(input_path)
    if p.suffix == ".sqlite":
        print(f"[1/4] Reusing existing {p.name}")
        return str(p)

    sqlite_path = str(p.with_suffix(".sqlite"))
    if os.path.exists(sqlite_path):
        print(f"[1/4] Reusing existing {Path(sqlite_path).name}")
        return sqlite_path

    print(f"[1/4] Exporting {p.name} → .sqlite ...")
    subprocess.run(
        [nsys_bin, "export", "--type=sqlite", f"--output={sqlite_path}", input_path],
        check=True,
    )
    return sqlite_path


def query_events(sqlite_path: str, start_ms=None, end_ms=None, top_kernels=None):
    """Query kernel and memcpy events from the sqlite database."""
    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row

    # Kernels
    try:
        kernels = [dict(r) for r in conn.execute(KERNEL_QUERY).fetchall()]
    except sqlite3.OperationalError:
        kernels = []

    # Memcpy
    try:
        memcpys = [dict(r) for r in conn.execute(MEMCPY_QUERY).fetchall()]
    except sqlite3.OperationalError:
        memcpys = []

    conn.close()

    if not kernels and not memcpys:
        print("  WARNING: No kernel or memcpy events found.")
        return [], [], 0

    # Find global time base
    all_starts = [k["start"] for k in kernels] + [m["start"] for m in memcpys]
    all_ends = [k["end"] for k in kernels] + [m["end"] for m in memcpys]
    t0 = min(all_starts)
    t_max = max(all_ends)

    # Apply time window filter
    if start_ms is not None:
        lo = t0 + int(start_ms * 1e6)
        kernels = [k for k in kernels if k["end"] >= lo]
        memcpys = [m for m in memcpys if m["end"] >= lo]
    if end_ms is not None:
        hi = t0 + int(end_ms * 1e6)
        kernels = [k for k in kernels if k["start"] <= hi]
        memcpys = [m for m in memcpys if m["start"] <= hi]

    # Top-N kernels by duration
    if top_kernels is not None:
        kernels.sort(key=lambda k: k["duration"], reverse=True)
        kernels = kernels[:top_kernels]
        kernels.sort(key=lambda k: k["start"])

    span_ms = (t_max - t0) / 1e6
    print(f"  {len(kernels)} kernels, {len(memcpys)} memcpy  ({0:.2f} – {span_ms:.2f} ms)")

    return kernels, memcpys, t0


def short_name(demangled: str) -> str:
    """Extract a readable short name from a demangled CUDA kernel name."""
    if not demangled:
        return "unknown_kernel"
    # Take the last qualified name before template args
    name = demangled.split("<")[0].split("(")[0]
    parts = name.split("::")
    # Return last 1-2 segments
    return "::".join(parts[-2:]) if len(parts) > 1 else parts[-1]


def fmt_bytes(b: int) -> str:
    if b >= 1 << 20:
        return f"{b / (1 << 20):.1f} MiB"
    if b >= 1 << 10:
        return f"{b / (1 << 10):.1f} KiB"
    return f"{b} B"


def write_csv(kernels, memcpys, t0, csv_path):
    """Write combined trace CSV."""
    header = [
        "event_type", "op_name", "full_name", "start_ns", "end_ns",
        "duration_ns", "start_ms", "end_ms", "duration_ms",
        "device_id", "stream_id", "bytes", "detail",
    ]

    rows = []
    for k in kernels:
        sname = short_name(k.get("demangledName", ""))
        grid = f"{k['gridX']}x{k['gridY']}x{k['gridZ']}" if k.get("gridX") else ""
        block = f"{k['blockX']}x{k['blockY']}x{k['blockZ']}" if k.get("blockX") else ""
        detail = f"grid={grid} block={block} regs={k.get('registersPerThread', '?')} shmem={fmt_bytes(k.get('sharedMem', 0))}"
        rows.append([
            "kernel", sname, k.get("demangledName", ""),
            k["start"], k["end"], k["duration"],
            f"{(k['start'] - t0) / 1e6:.3f}",
            f"{(k['end'] - t0) / 1e6:.3f}",
            f"{k['duration'] / 1e6:.3f}",
            k.get("deviceId", ""), k.get("streamId", ""), 0, detail,
        ])

    for m in memcpys:
        ck = MEMCPY_KIND.get(m["copyKind"], str(m["copyKind"]))
        sk = MEM_KIND.get(m["srcKind"], str(m["srcKind"]))
        dk = MEM_KIND.get(m["dstKind"], str(m["dstKind"]))
        op = f"{ck} ({fmt_bytes(m['bytes'])})"
        detail = f"{sk} → {dk}"
        rows.append([
            "memcpy", op, ck,
            m["start"], m["end"], m["duration"],
            f"{(m['start'] - t0) / 1e6:.3f}",
            f"{(m['end'] - t0) / 1e6:.3f}",
            f"{m['duration'] / 1e6:.3f}",
            m.get("deviceId", ""), m.get("streamId", ""), m["bytes"], detail,
        ])

    rows.sort(key=lambda r: int(r[3]))  # sort by start_ns

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"[3/4] Writing CSV: {csv_path} ({len(rows)} rows)")
    return rows


def write_html(kernels, memcpys, t0, html_path):
    """Build a self-contained interactive HTML timeline."""
    events = []

    for k in kernels:
        sname = short_name(k.get("demangledName", ""))
        events.append({
            "type": "kernel",
            "name": sname,
            "full": k.get("demangledName", ""),
            "start": (k["start"] - t0) / 1e6,
            "end": (k["end"] - t0) / 1e6,
            "dur": k["duration"] / 1e6,
            "stream": k.get("streamId", 0),
            "device": k.get("deviceId", 0),
        })

    for m in memcpys:
        ck = MEMCPY_KIND.get(m["copyKind"], str(m["copyKind"]))
        events.append({
            "type": "memcpy",
            "name": f"{ck} ({fmt_bytes(m['bytes'])})",
            "full": f"{MEM_KIND.get(m['srcKind'], '?')} → {MEM_KIND.get(m['dstKind'], '?')}",
            "start": (m["start"] - t0) / 1e6,
            "end": (m["end"] - t0) / 1e6,
            "dur": m["duration"] / 1e6,
            "stream": m.get("streamId", 0),
            "device": m.get("deviceId", 0),
            "bytes": m["bytes"],
        })

    events.sort(key=lambda e: e["start"])
    events_json = json.dumps(events)

    # Compute summary stats for the HTML
    kernel_events = [e for e in events if e["type"] == "kernel"]
    memcpy_events = [e for e in events if e["type"] == "memcpy"]

    # Top 10 kernels by duration
    top_kernels = sorted(kernel_events, key=lambda e: e["dur"], reverse=True)[:10]

    # Memcpy breakdown
    htod = [e for e in memcpy_events if "HtoD" in e["name"]]
    dtoh = [e for e in memcpy_events if "DtoH" in e["name"]]
    dtod = [e for e in memcpy_events if "DtoD" in e["name"]]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Nsight Systems Trace Visualization</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace; background: #1a1a2e; color: #e0e0e0; }}
.header {{ background: #16213e; padding: 16px 24px; border-bottom: 2px solid #0f3460; }}
.header h1 {{ font-size: 18px; color: #e94560; }}
.header .stats {{ font-size: 13px; color: #999; margin-top: 4px; }}
.controls {{ padding: 12px 24px; background: #16213e; border-bottom: 1px solid #0f3460; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
.controls label {{ font-size: 12px; color: #999; }}
.controls input, .controls select {{ background: #1a1a2e; border: 1px solid #0f3460; color: #e0e0e0; padding: 4px 8px; border-radius: 4px; font-size: 12px; }}
.controls button {{ background: #0f3460; color: #e0e0e0; border: none; padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 12px; }}
.controls button:hover {{ background: #e94560; }}
.main {{ display: flex; height: calc(100vh - 140px); }}
.timeline {{ flex: 1; overflow: auto; position: relative; }}
canvas {{ display: block; }}
.sidebar {{ width: 360px; background: #16213e; border-left: 1px solid #0f3460; overflow-y: auto; padding: 12px; }}
.sidebar h3 {{ font-size: 13px; color: #e94560; margin-bottom: 8px; padding-bottom: 4px; border-bottom: 1px solid #0f3460; }}
.sidebar table {{ width: 100%; font-size: 11px; border-collapse: collapse; margin-bottom: 16px; }}
.sidebar th {{ text-align: left; color: #999; padding: 3px 6px; }}
.sidebar td {{ padding: 3px 6px; border-top: 1px solid #0f3460; }}
.sidebar .dur {{ color: #e94560; font-weight: bold; }}
.tooltip {{ position: fixed; background: #16213e; border: 1px solid #e94560; padding: 8px 12px; border-radius: 6px; font-size: 11px; pointer-events: none; z-index: 100; max-width: 500px; display: none; }}
.tooltip .tt-name {{ color: #e94560; font-weight: bold; }}
.tooltip .tt-row {{ color: #ccc; margin-top: 2px; }}
.legend {{ display: flex; gap: 16px; font-size: 11px; margin-left: auto; }}
.legend span {{ display: flex; align-items: center; gap: 4px; }}
.legend .dot {{ width: 10px; height: 10px; border-radius: 2px; }}
</style>
</head>
<body>
<div class="header">
    <h1>Nsight Systems Trace — Qwen3-VL-2B Benchmark</h1>
    <div class="stats">
        {len(kernel_events)} kernels | {len(memcpy_events)} memcpy events |
        Total: {events[-1]["end"]:.2f} ms
    </div>
</div>
<div class="controls">
    <label>From (ms):</label><input type="number" id="startMs" step="0.1" value="0">
    <label>To (ms):</label><input type="number" id="endMs" step="0.1" value="{events[-1]['end']:.1f}">
    <button onclick="applyZoom()">Zoom</button>
    <button onclick="resetZoom()">Reset</button>
    <label>Filter:</label>
    <select id="filterType">
        <option value="all">All events</option>
        <option value="kernel">Kernels only</option>
        <option value="memcpy">Memcpy only</option>
    </select>
    <div class="legend">
        <span><div class="dot" style="background:#4fc3f7"></div> Kernel</span>
        <span><div class="dot" style="background:#e94560"></div> HtoD</span>
        <span><div class="dot" style="background:#66bb6a"></div> DtoH</span>
        <span><div class="dot" style="background:#ffa726"></div> DtoD</span>
    </div>
</div>
<div class="main">
    <div class="timeline" id="timelineContainer">
        <canvas id="canvas"></canvas>
    </div>
    <div class="sidebar">
        <h3>Top 10 Kernels (by duration)</h3>
        <table>
            <tr><th>#</th><th>Kernel</th><th>ms</th></tr>
            {"".join(f'<tr><td>{i+1}</td><td title="{html.escape(k["full"])}">{html.escape(k["name"][:40])}</td><td class="dur">{k["dur"]:.3f}</td></tr>' for i, k in enumerate(top_kernels))}
        </table>
        <h3>Memcpy Summary</h3>
        <table>
            <tr><th>Direction</th><th>Count</th><th>Total ms</th><th>Total Size</th></tr>
            <tr><td>HtoD</td><td>{len(htod)}</td><td class="dur">{sum(e['dur'] for e in htod):.3f}</td><td>{fmt_bytes(sum(e.get('bytes',0) for e in htod))}</td></tr>
            <tr><td>DtoH</td><td>{len(dtoh)}</td><td class="dur">{sum(e['dur'] for e in dtoh):.3f}</td><td>{fmt_bytes(sum(e.get('bytes',0) for e in dtoh))}</td></tr>
            <tr><td>DtoD</td><td>{len(dtod)}</td><td class="dur">{sum(e['dur'] for e in dtod):.3f}</td><td>{fmt_bytes(sum(e.get('bytes',0) for e in dtod))}</td></tr>
        </table>
        <h3>Event Detail</h3>
        <div id="detail" style="font-size:11px; color:#999;">Hover over an event on the timeline.</div>
    </div>
</div>
<div class="tooltip" id="tooltip"></div>

<script>
const ALL_EVENTS = {events_json};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const container = document.getElementById('timelineContainer');
const tooltip = document.getElementById('tooltip');
const detail = document.getElementById('detail');

let viewStart = 0;
let viewEnd = ALL_EVENTS.length ? ALL_EVENTS[ALL_EVENTS.length-1].end : 1;
let events = ALL_EVENTS;
let streamMap = {{}};
let rowHeight = 18;
let headerHeight = 30;
let leftGutter = 60;

function getColor(e) {{
    if (e.type === 'kernel') return '#4fc3f7';
    if (e.name.includes('HtoD')) return '#e94560';
    if (e.name.includes('DtoH')) return '#66bb6a';
    if (e.name.includes('DtoD')) return '#ffa726';
    return '#ab47bc';
}}

function filteredEvents() {{
    const ft = document.getElementById('filterType').value;
    let evts = ALL_EVENTS;
    if (ft !== 'all') evts = evts.filter(e => e.type === ft);
    return evts.filter(e => e.end >= viewStart && e.start <= viewEnd);
}}

function buildStreamMap(evts) {{
    const streams = [...new Set(evts.map(e => e.stream))].sort((a,b) => a-b);
    streamMap = {{}};
    streams.forEach((s, i) => streamMap[s] = i);
    return streams.length;
}}

function draw() {{
    events = filteredEvents();
    const numStreams = buildStreamMap(events);
    const height = headerHeight + numStreams * rowHeight + 20;
    const width = container.clientWidth;
    canvas.width = width * devicePixelRatio;
    canvas.height = height * devicePixelRatio;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.scale(devicePixelRatio, devicePixelRatio);

    const plotW = width - leftGutter;
    const msRange = viewEnd - viewStart;

    // Background
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, width, height);

    // Time axis ticks
    ctx.fillStyle = '#555';
    ctx.font = '10px monospace';
    const tickCount = Math.min(20, Math.max(5, Math.floor(plotW / 80)));
    for (let i = 0; i <= tickCount; i++) {{
        const ms = viewStart + (msRange * i / tickCount);
        const x = leftGutter + (ms - viewStart) / msRange * plotW;
        ctx.fillRect(x, headerHeight - 5, 1, height);
        ctx.fillStyle = '#888';
        ctx.fillText(ms.toFixed(ms < 10 ? 2 : 1) + ' ms', x + 3, headerHeight - 8);
        ctx.fillStyle = '#333';
    }}

    // Stream labels
    ctx.fillStyle = '#888';
    ctx.font = '10px monospace';
    for (const [stream, idx] of Object.entries(streamMap)) {{
        const y = headerHeight + idx * rowHeight + rowHeight / 2 + 4;
        ctx.fillText('S' + stream, 4, y);
    }}

    // Events
    for (const e of events) {{
        const row = streamMap[e.stream] ?? 0;
        const x = leftGutter + (e.start - viewStart) / msRange * plotW;
        const w = Math.max(1, (e.end - e.start) / msRange * plotW);
        const y = headerHeight + row * rowHeight + 2;
        const h = rowHeight - 4;
        ctx.fillStyle = getColor(e);
        ctx.fillRect(x, y, w, h);

        // Label if wide enough
        if (w > 40) {{
            ctx.fillStyle = '#000';
            ctx.font = '9px monospace';
            ctx.fillText(e.name.substring(0, Math.floor(w / 5.5)), x + 2, y + h - 3);
        }}
    }}
}}

// Tooltip
canvas.addEventListener('mousemove', (ev) => {{
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const my = ev.clientY - rect.top;
    const plotW = canvas.clientWidth - leftGutter;
    const msRange = viewEnd - viewStart;
    const msAtCursor = viewStart + (mx - leftGutter) / plotW * msRange;

    let hit = null;
    for (const e of events) {{
        const row = streamMap[e.stream] ?? 0;
        const y = headerHeight + row * rowHeight + 2;
        const h = rowHeight - 4;
        if (my >= y && my <= y + h && msAtCursor >= e.start && msAtCursor <= e.end) {{
            hit = e;
            break;
        }}
    }}

    if (hit) {{
        tooltip.style.display = 'block';
        tooltip.style.left = (ev.clientX + 12) + 'px';
        tooltip.style.top = (ev.clientY + 12) + 'px';
        let inner = `<div class="tt-name">${{hit.name}}</div>`;
        inner += `<div class="tt-row">Duration: ${{hit.dur.toFixed(4)}} ms</div>`;
        inner += `<div class="tt-row">Time: ${{hit.start.toFixed(3)}} – ${{hit.end.toFixed(3)}} ms</div>`;
        inner += `<div class="tt-row">Stream: ${{hit.stream}} | Device: ${{hit.device}}</div>`;
        if (hit.bytes) inner += `<div class="tt-row">Size: ${{(hit.bytes/(1<<20)).toFixed(2)}} MiB</div>`;
        if (hit.full) inner += `<div class="tt-row" style="color:#666;max-width:400px;word-break:break-all">${{hit.full}}</div>`;
        tooltip.innerHTML = inner;

        detail.innerHTML = inner;
    }} else {{
        tooltip.style.display = 'none';
    }}
}});

canvas.addEventListener('mouseleave', () => {{ tooltip.style.display = 'none'; }});

// Zoom via scroll
canvas.addEventListener('wheel', (ev) => {{
    ev.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = ev.clientX - rect.left;
    const plotW = canvas.clientWidth - leftGutter;
    const frac = (mx - leftGutter) / plotW;
    const msRange = viewEnd - viewStart;
    const factor = ev.deltaY > 0 ? 1.2 : 1/1.2;
    const newRange = msRange * factor;
    const anchor = viewStart + frac * msRange;
    viewStart = anchor - frac * newRange;
    viewEnd = anchor + (1 - frac) * newRange;
    draw();
}}, {{ passive: false }});

// Pan via drag
let dragging = false, dragStartX = 0, dragViewStart = 0;
canvas.addEventListener('mousedown', (ev) => {{
    dragging = true; dragStartX = ev.clientX; dragViewStart = viewStart;
}});
canvas.addEventListener('mousemove', (ev) => {{
    if (!dragging) return;
    const dx = ev.clientX - dragStartX;
    const plotW = canvas.clientWidth - leftGutter;
    const msRange = viewEnd - viewStart;
    const shift = -dx / plotW * msRange;
    const newStart = dragViewStart + shift;
    viewEnd = newStart + msRange;
    viewStart = newStart;
    draw();
}});
canvas.addEventListener('mouseup', () => {{ dragging = false; }});

function applyZoom() {{
    viewStart = parseFloat(document.getElementById('startMs').value) || 0;
    viewEnd = parseFloat(document.getElementById('endMs').value) || viewEnd;
    draw();
}}
function resetZoom() {{
    viewStart = 0;
    viewEnd = ALL_EVENTS.length ? ALL_EVENTS[ALL_EVENTS.length-1].end : 1;
    document.getElementById('startMs').value = '0';
    document.getElementById('endMs').value = viewEnd.toFixed(1);
    draw();
}}

document.getElementById('filterType').addEventListener('change', draw);
window.addEventListener('resize', draw);
draw();
</script>
</body>
</html>"""

    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"[4/4] Building HTML: {html_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Nsight Systems trace")
    parser.add_argument("--input", required=True, help=".nsys-rep or .sqlite file")
    parser.add_argument("--start-ms", type=float, default=None, help="Crop start (ms)")
    parser.add_argument("--end-ms", type=float, default=None, help="Crop end (ms)")
    parser.add_argument("--top-kernels", type=int, default=None, help="Only top N kernels")
    parser.add_argument("--nsys-bin", default=DEFAULT_NSYS, help="Path to nsys binary")
    args = parser.parse_args()

    sqlite_path = ensure_sqlite(args.input, args.nsys_bin)

    print("[2/4] Querying kernel + memcpy events ...")
    kernels, memcpys, t0 = query_events(
        sqlite_path, args.start_ms, args.end_ms, args.top_kernels
    )

    base = Path(sqlite_path).with_suffix("")
    csv_path = str(base) + "_trace.csv"
    html_path = str(base) + "_trace.html"

    write_csv(kernels, memcpys, t0, csv_path)
    write_html(kernels, memcpys, t0, html_path)

    print("Done.")


if __name__ == "__main__":
    main()
