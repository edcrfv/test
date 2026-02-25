# plot_3 — Kernel Compute vs Memory Copy (2260–2400ms window)

Two-row Plotly timeline showing GPU activity during Qwen3-VL-2B inference.

## Layout

```
┌─────────────────────────────────────────────────────────────┐
│ Kernel Compute  ██ ██ ██████ ██ ██ ██ ██████ ██ ██ ██       │  ← CUDA kernels (blue)
│ Memory Copy         ████         ████████                    │  ← memcpy (colored by direction)
└─────────────────────────────────────────────────────────────┘
                  ──────── Time (ms) ────────►
```

## Legend

| Color | Meaning |
|-------|---------|
| Blue (`#4fc3f7`) | Compute kernel (GEMM, attention, elementwise) |
| Red (`#ef5350`) | HtoD — CPU RAM → GPU VRAM (weight loading) |
| Green (`#66bb6a`) | DtoH — GPU VRAM → CPU RAM (sync signals) |
| Orange (`#ffa726`) | DtoD — GPU internal memory reshuffling |
| Purple (`#ab47bc`) | Peer — device-to-device (NVLink/PCIe) |

## What this window shows

The 2260–2400ms window captures **~6 transformer layer forward passes** during autoregressive token generation. Each layer follows a repeating pattern:

1. **flash_fwd_splitkv_kernel** — attention computation
2. **HtoD 8 MiB** — next layer's weights streamed from CPU
3. **elementwise / vectorized ops** — activations, norms, gating
4. **HtoD 24 MiB × 3** — weight matrix shards (gate, up, down projections)
5. **cutlass::Kernel2 / cublas gemv** — matrix multiply

## Related files

- `selected_csv/kernels_2260-2400ms.csv` — all 1248 kernels with timing, grid/block config
- `selected_csv/memcpy_2260-2400ms.csv` — all 65 memcpy with size, bandwidth
- `selected_csv/memcpy_e2e_2260-2400ms.csv` — CPU call vs GPU DMA timing per transfer
- `selected_csv/kernel_memcpy_pairs_2260-2400ms.csv` — which kernel ran before/after each memcpy

## Regenerate

```bash
python scripts/plot_nsys_trace.py \
    --sqlite nsys_reports/qwen3vl_2b_20260224_215843.sqlite \
    --start-ms 2260 --end-ms 2400 \
    --output plot_3/trace_2260-2400ms.html
```
