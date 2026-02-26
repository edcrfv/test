# Qwen3-VL-2B Inference Profiling

GPU profiling and visualization of **Qwen3-VL-2B-Instruct** inference on an RTX 5090 using NVIDIA Nsight Systems.

Compares eager execution vs `torch.compile` — kernel compute, memory copy timing, and CPU↔GPU transfer analysis.

## Results

| | Eager | torch.compile |
|---|---|---|
| Avg latency | 2.327 s | **1.726 s** |
| Throughput | 55.01 tok/s | **74.16 tok/s (+35%)** |
| Min latency | 2.261 s | 1.703 s |
| Max latency | 2.392 s | 1.774 s |

## Repository Structure

```
.
├── benchmark.py                  # Eager inference benchmark
├── benchmark_compiled.py         # torch.compile inference benchmark
│
├── scripts/
│   ├── run_benchmark_nsys.sh     # nsys profiling wrapper
│   ├── export_nsys_memcpy_csv.py # .sqlite → memcpy CSV
│   ├── analyze_memcpy.py         # E2E memcpy analysis (CPU call vs GPU DMA)
│   ├── dump_window_csv.py        # Dump events from a time window to CSV
│   ├── dump_kernel_memcpy_pairs.py # Which kernel ran before/after each memcpy
│   ├── plot_nsys_trace.py        # Plotly 2-row timeline (kernel vs memcpy)
│   ├── plot_memcpy_e2e.py        # Plotly CPU↔GPU memcpy correlation
│   └── visualize_nsys_trace.py   # Self-contained HTML timeline (canvas)
│
├── nsys_reports/                 # Raw profiles + exports (LFS-tracked)
│   ├── qwen3vl_2b_20260224_*.nsys-rep/.sqlite   # Eager runs
│   └── qwen3vl_2b_compiled_*.nsys-rep/.sqlite    # Compiled run
│
├── plot_2/                       # Full inference window plots
│   ├── model_loading.html        #   0–1000ms — weight loading phase
│   └── inference.html            #   1500–4000ms — token generation
│
├── plot_3/                       # Zoomed window (2260–2400ms, eager)
│   ├── trace_2260-2400ms.html
│   ├── trimmed.html
│   └── README.md
│
├── plot_select/                  # Selected window plot
│   └── trace_2260-2400ms.html
│
├── plot_compiled/                # torch.compile plots (2260–2400ms)
│   ├── trace_2260-2400ms.html
│   └── trimmed.html
│
├── csv_run_3/                    # Eager CSV dumps (2260–2400ms)
│   ├── trimmed.csv               #   3-column: engine, start_time, end_time
│   ├── kernels_2260-2400ms.csv
│   ├── memcpy_2260-2400ms.csv
│   ├── memcpy_e2e_2260-2400ms.csv
│   └── kernel_memcpy_pairs_2260-2400ms.csv
│
├── csv_compiled/                 # torch.compile CSV dumps (2260–2400ms)
│   ├── trimmed.csv
│   ├── kernels_2260-2400ms.csv
│   ├── memcpy_2260-2400ms.csv
│   ├── memcpy_e2e_2260-2400ms.csv
│   └── kernel_memcpy_pairs_2260-2400ms.csv
│
└── selected_csv/                 # Earlier CSV dumps (same window)
```

## Setup

```bash
# Create conda environment
conda create -n qwen32b python=3.11 -y
conda activate qwen32b

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate qwen-vl-utils plotly pandas
```

Nsight Systems was installed without root by extracting the deb to `~/nsight-systems/`.

## Usage

### Run benchmarks

```bash
# Eager
python benchmark.py

# torch.compile
python benchmark_compiled.py
```

### Profile with nsys

```bash
export PATH="/home/a1/nsight-systems/opt/nvidia/nsight-systems/2026.1.1/bin:$PATH"

# Profile and export
nsys profile --trace=cuda,nvtx,osrt --output=nsys_reports/my_run python benchmark.py
nsys export --type=sqlite --output=nsys_reports/my_run.sqlite nsys_reports/my_run.nsys-rep
```

### Generate plots

```bash
# 2-row timeline (Kernel Compute vs Memory Copy)
python scripts/plot_nsys_trace.py --sqlite nsys_reports/*.sqlite \
    --start-ms 2260 --end-ms 2400 --output plot_3/trace.html

# CPU↔GPU memcpy correlation
python scripts/plot_memcpy_e2e.py --sqlite nsys_reports/*.sqlite \
    --end-ms 1000 --min-bytes 4096

# Memcpy analysis (text summary + CSV)
python scripts/analyze_memcpy.py --sqlite nsys_reports/*.sqlite
```

### Dump CSVs

```bash
# All events in a time window
python scripts/dump_window_csv.py --sqlite nsys_reports/*.sqlite \
    --t1 2260 --t2 2400 --out-dir my_csv

# Kernel↔memcpy pairs (which kernel before/after each copy)
python scripts/dump_kernel_memcpy_pairs.py --sqlite nsys_reports/*.sqlite \
    --t1 2260 --t2 2400 --out-dir my_csv
```

## Key Findings

**Memory transfers during model loading (0–1000ms):**
- 3.97 GiB HtoD (CPU→GPU) at 10–15 GB/s over PCIe
- Source memory is **pageable** (not pinned) — large staging overhead
- Launch overhead often exceeds actual DMA time

**During inference (1000ms+):**
- Zero HtoD — all weights on GPU
- Only Peer (device-to-device) reshuffling and tiny DtoH sync signals
- Repeating per-layer pattern: attention → HtoD weights → elementwise → GEMM

**torch.compile impact:**
- 35% throughput improvement (55 → 74 tok/s)
- More kernel fusions — 17,944 kernels vs 1,248 in same 140ms window
- Kernels are smaller/faster individually but more numerous due to graph capture

## Hardware

- GPU: NVIDIA GeForce RTX 5090 (32 GB GDDR7)
- Driver: 580.95.05 / CUDA 13.0
- PyTorch: 2.10.0+cu128
- Transformers: 5.2.0
