from onnxruntime.quantization import quantize_dynamic, QuantType
from huggingface_hub import HfApi, create_repo
import os
import sys
import io

# Fix window encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

IN_DIR = "onnx_layers_1.7b"
OUT_DIR = "onnx_layers_1.7b_int8"
REPO_ID = "powerpudu/Web-Petals-SmolLM2-1.7B-q8"

if not os.path.exists(IN_DIR):
    print(f"Error: {IN_DIR} not found.")
    sys.exit(1)

os.makedirs(OUT_DIR, exist_ok=True)
files = [f for f in os.listdir(IN_DIR) if f.endswith(".onnx")]

print("=== 1. Quantizing Model to QInt8 ===")
for idx, f in enumerate(sorted(files)):
    in_path = os.path.join(IN_DIR, f)
    out_path = os.path.join(OUT_DIR, f)
    if not os.path.exists(out_path):
        print(f"[{idx+1}/{len(files)}] Quantizando {f}...")
        quantize_dynamic(in_path, out_path, weight_type=QuantType.QInt8)
    else:
        print(f"[{idx+1}/{len(files)}] Skipped {f} (ya existe)")

total_mb = sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in files) / 1024 / 1024
print(f"Total Model Size: {total_mb:.1f} MB")

print("\n=== 2. Uploading to HuggingFace ===")
api = HfApi()
try:
    create_repo(REPO_ID, repo_type="model", exist_ok=True)
    print(f"Repo {REPO_ID} creado/accesible.")
except Exception as e:
    print(f"Error con el repo: {e}")

for idx, f in enumerate(sorted(files)):
    out_path = os.path.join(OUT_DIR, f)
    print(f"[{idx+1}/{len(files)}] Uploading {f} descde local ({os.path.getsize(out_path)/1024/1024:.1f} MB)...", end=" ", flush=True)
    try:
        api.upload_file(
            path_or_fileobj=out_path,
            path_in_repo=f,
            repo_id=REPO_ID,
            repo_type="model"
        )
        print("✅")
    except Exception as e:
        print(f"❌ Error: {e}")

# Upload README
readme = f"""---
tags:
  - onnx
  - web-petals
  - distributed-inference
  - smollm2
  - int8
license: apache-2.0
---

# Web-Petals SmolLM2-1.7B ONNX Layers (QInt8)

SmolLM2-1.7B-Instruct split into individual ONNX layers and **dynamically quantized to INT8** for distributed P2P inference in WebGPU/WASM.

- **Total Size**: ~{total_mb:.1f} MB
- **Layer Size**: ~64 MB
- **Precision**: INT8 (weights only)
"""
readme_path = os.path.join(OUT_DIR, "README.md")
with open(readme_path, "w", encoding="utf-8") as rf:
    rf.write(readme)

api.upload_file(path_or_fileobj=readme_path, path_in_repo="README.md", repo_id=REPO_ID, repo_type="model")

print("\n✅ Completado!")
