#!/usr/bin/env python3
"""Converts fused HuggingFace Safetensors (`posttrain/fused_hf`) to F16 GGUF and quantizes to `Q4_K_M`.

Produces `posttrain/models/shaanpatel-cv-pico-q4_k_m.gguf`, which `spatel --build-pico-model`
auto-discovers and imports into Ollama's GGUF runner.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FUSED_DIR = ROOT / "fused_hf"
MODELS_DIR = ROOT / "models"
CACHE_DIR = ROOT / ".cache" / "llama_cpp_b10050"

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_TAG = "b10050"


def ensure_converter() -> Path:
    converter = CACHE_DIR / "convert_hf_to_gguf.py"
    if converter.is_file() and (CACHE_DIR / "conversion").is_dir():
        return converter

    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.parent.mkdir(parents=True, exist_ok=True)

    print(f"Fetching sparse llama.cpp converter ({LLAMA_CPP_TAG}) -> {CACHE_DIR}", flush=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "--branch",
            LLAMA_CPP_TAG,
            LLAMA_CPP_REPO,
            str(CACHE_DIR),
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(CACHE_DIR),
            "sparse-checkout",
            "set",
            "--no-cone",
            "/convert_hf_to_gguf.py",
            "/conversion/**",
            "/gguf-py/**",
        ],
        check=True,
    )
    return converter


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and quantize fused Safetensors to Q4_K_M GGUF.")
    parser.add_argument("--fused-dir", type=Path, default=FUSED_DIR)
    parser.add_argument(
        "--out-gguf",
        type=Path,
        default=MODELS_DIR / "shaanpatel-cv-pico-q4_k_m.gguf",
    )
    parser.add_argument("--quant-type", default="Q4_K_M")
    parser.add_argument(
        "--llama-quantize",
        default=shutil.which("llama-quantize") or "/opt/homebrew/bin/llama-quantize",
    )
    args = parser.parse_args()

    if not (args.fused_dir / "model.safetensors").is_file() and not (
        args.fused_dir / "model.safetensors.index.json"
    ).is_file():
        raise FileNotFoundError(f"Missing Safetensors weights in {args.fused_dir}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    converter = ensure_converter()
    f16_gguf = MODELS_DIR / "shaanpatel-cv-pico-f16.gguf"

    print(f"[1/2] Converting {args.fused_dir} -> {f16_gguf}", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(converter),
            str(args.fused_dir),
            "--outfile",
            str(f16_gguf),
            "--outtype",
            "f16",
        ],
        check=True,
    )

    print(f"[2/2] Quantizing {f16_gguf} -> {args.out_gguf} ({args.quant_type})", flush=True)
    subprocess.run(
        [
            args.llama_quantize,
            str(f16_gguf),
            str(args.out_gguf),
            args.quant_type,
        ],
        check=True,
    )

    if f16_gguf.is_file() and args.out_gguf.is_file():
        f16_gguf.unlink()

    size_mb = round(args.out_gguf.stat().st_size / (1024 * 1024), 1)
    print(f"Exported quantized GGUF: {args.out_gguf} ({size_mb} MB)", flush=True)


if __name__ == "__main__":
    main()
