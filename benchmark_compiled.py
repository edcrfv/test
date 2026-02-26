import time
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# =========================
# Config
# =========================
MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
IMAGE_PATH = "test.jpg"

WARMUP_RUNS = 3
BENCH_RUNS = 10
MAX_NEW_TOKENS = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# =========================
# Load model & processor
# =========================
print("Loading model...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None
).eval()

if DEVICE == "cpu":
    model = model.to(DEVICE)

# =========================
# torch.compile
# =========================
print("Compiling model with torch.compile...")
model = torch.compile(model, mode="reduce-overhead")

# =========================
# Prepare input
# =========================
image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }
]

text = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = processor(
    text=[text],
    images=[image],
    return_tensors="pt"
).to(DEVICE)

# =========================
# Warm-up (includes compilation)
# =========================
print(f"\nWarming up ({WARMUP_RUNS} runs, includes JIT compilation)...")

for i in range(WARMUP_RUNS):
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"  Warmup {i+1}: {t1 - t0:.3f}s")

# =========================
# Benchmark
# =========================
print(f"\nBenchmarking ({BENCH_RUNS} runs)...")

latencies = []
token_counts = []

for i in range(BENCH_RUNS):
    start = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS
        )

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    latency = end - start
    latencies.append(latency)

    new_tokens = outputs.shape[-1] - inputs["input_ids"].shape[-1]
    token_counts.append(new_tokens)

    print(f"Run {i+1}: {latency:.3f}s, tokens: {new_tokens}")

# =========================
# Results
# =========================
avg_latency = sum(latencies) / len(latencies)
avg_tokens = sum(token_counts) / len(token_counts)
throughput = avg_tokens / avg_latency

print("\n===== RESULTS (torch.compile) =====")
print(f"Device: {DEVICE}")
print(f"Avg latency: {avg_latency:.3f} s")
print(f"Avg tokens generated: {avg_tokens:.1f}")
print(f"Throughput: {throughput:.2f} tokens/sec")
print(f"Min latency: {min(latencies):.3f} s")
print(f"Max latency: {max(latencies):.3f} s")
