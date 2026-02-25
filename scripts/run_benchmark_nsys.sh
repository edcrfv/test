#!/usr/bin/env bash
set -euo pipefail

# ─── Nsight Systems profiling wrapper for benchmark.py ───
# Usage:
#   bash scripts/run_benchmark_nsys.sh [--output_dir DIR] [-- extra python args]
#
# Wraps benchmark.py with:  nsys profile --trace=cuda,nvtx,osrt
# Produces: nsys_reports/<timestamp>.nsys-rep  →  .sqlite  →  _memcpy_trace.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
NSYS_BIN="${NSYS_BIN:-/home/a1/nsight-systems/opt/nvidia/nsight-systems/2026.1.1/bin/nsys}"
OUTPUT_DIR="${PROJECT_DIR}/nsys_reports"
PYTHON_ARGS=()

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --) shift; PYTHON_ARGS+=("$@"); break ;;
        *) PYTHON_ARGS+=("$1"); shift ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_BASE="${OUTPUT_DIR}/qwen3vl_2b_${TIMESTAMP}"
REPORT_FILE="${REPORT_BASE}.nsys-rep"

echo "=== Nsight Systems Profiling ==="
echo "Output dir : $OUTPUT_DIR"
echo "Report base: $REPORT_BASE"
echo ""

# ─── Step 1: Profile ───
echo "[1/3] Running benchmark.py under nsys profile ..."
"$NSYS_BIN" profile \
    --trace=cuda,nvtx,osrt \
    --output="$REPORT_BASE" \
    --force-overwrite=true \
    --stats=false \
    python "${PROJECT_DIR}/benchmark.py" "${PYTHON_ARGS[@]}"

echo ""
echo "[1/3] Done → $REPORT_FILE"

# ─── Step 2: Export to SQLite ───
SQLITE_FILE="${REPORT_BASE}.sqlite"
echo "[2/3] Exporting to SQLite ..."
"$NSYS_BIN" export --type=sqlite --output="$SQLITE_FILE" "$REPORT_FILE"
echo "[2/3] Done → $SQLITE_FILE"

# ─── Step 3: Extract memcpy trace CSV ───
CSV_FILE="${REPORT_BASE}_memcpy_trace.csv"
echo "[3/3] Extracting memcpy trace ..."
python3 "${SCRIPT_DIR}/export_nsys_memcpy_csv.py" "$SQLITE_FILE" "$CSV_FILE"
echo "[3/3] Done → $CSV_FILE"

echo ""
echo "=== Profiling Complete ==="
echo "  .nsys-rep : $REPORT_FILE"
echo "  .sqlite   : $SQLITE_FILE"
echo "  memcpy CSV: $CSV_FILE"
