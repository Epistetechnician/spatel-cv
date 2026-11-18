#!/usr/bin/env python3
"""End-to-end LoRA SFT post-training and FP16 fusion for shaanpatel-cv-pico on Apple Silicon (MLX).

Usage (from spatel-cv/):
    ./posttrain/.venv/bin/python posttrain/run_posttrain.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ADAPTER_DIR = ROOT / "adapters"
FUSED_DIR = ROOT / "fused_hf"
REPORT_PATH = ROOT / "train_report.json"

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def detect_cli_module(python_bin: str, subcmd: str) -> tuple[list[str], str]:
    """Supports both `python -m mlx_lm <subcmd>` (newer mlx-lm) and `python -m mlx_lm.<subcmd>`."""
    probe = subprocess.run(
        [python_bin, "-m", "mlx_lm", subcmd, "--help"],
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        return [python_bin, "-m", "mlx_lm", subcmd], probe.stdout
    probe_legacy = subprocess.run(
        [python_bin, "-m", f"mlx_lm.{subcmd}", "--help"],
        capture_output=True,
        text=True,
    )
    if probe_legacy.returncode == 0:
        return [python_bin, "-m", f"mlx_lm.{subcmd}"], probe_legacy.stdout
    raise RuntimeError(f"Could not locate mlx_lm {subcmd} entrypoint:\n{probe.stderr}\n{probe_legacy.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-train shaanpatel-cv-pico weights with MLX LoRA SFT.")
    parser.add_argument("--hf-model", default=DEFAULT_HF_MODEL)
    parser.add_argument("--iters", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-scale", type=float, default=2.0)
    parser.add_argument("--num-layers", type=int, default=-1)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    # 1. Ensure deterministic dataset is built and leak-checked
    subprocess.run([sys.executable, str(ROOT / "build_dataset.py")], check=True)

    if ADAPTER_DIR.exists():
        shutil.rmtree(ADAPTER_DIR)
    if FUSED_DIR.exists():
        shutil.rmtree(FUSED_DIR)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    lora_cmd_prefix, lora_help = detect_cli_module(sys.executable, "lora")
    fuse_cmd_prefix, fuse_help = detect_cli_module(sys.executable, "fuse")

    warmup_steps = 20
    decay_steps = max(20, args.iters - warmup_steps)
    end_lr = round(args.learning_rate * 0.1, 8)

    # Note: In mlx-lm, `lora_parameters.scale` is the direct multiplier (alpha / r = 32 / 16 = 2.0),
    # NOT raw PEFT `lora_alpha`. Setting scale=2.0 with cosine warmup prevents early BF16 overshoot.
    config_yaml = ROOT / "lora_config.yaml"
    config_lines = [
        f'model: "{args.hf_model}"',
        "train: true",
        'optimizer: "adamw"',
        f'data: "{DATA_DIR}"',
        "seed: 20261118",
        f"num_layers: {args.num_layers}",
        f"batch_size: {args.batch_size}",
        f"iters: {args.iters}",
        "val_batches: -1",
        f"learning_rate: {args.learning_rate}",
        "lr_schedule:",
        '  name: "cosine_decay"',
        f"  warmup: {warmup_steps}",
        f"  arguments: [{args.learning_rate}, {decay_steps}, {end_lr}]",
        "steps_per_report: 20",
        "steps_per_eval: 50",
        f'adapter_path: "{ADAPTER_DIR}"',
        "save_every: 130",
        f"max_seq_length: {args.max_seq_length}",
        "grad_checkpoint: true",
        "lora_parameters:",
        f"  rank: {args.lora_rank}",
        "  dropout: 0.05",
        f"  scale: {args.lora_scale}",
    ]
    if "--mask-prompt" in lora_help or "mask_prompt" in lora_help:
        config_lines.append("mask_prompt: true")
    config_yaml.write_text("\n".join(config_lines) + "\n", encoding="utf-8")

    lora_cmd = [*lora_cmd_prefix, "--config", str(config_yaml)]

    print(f"[1/2] Running MLX LoRA SFT: {' '.join(lora_cmd)}", flush=True)
    t0 = time.time()
    train_proc = subprocess.run(lora_cmd, text=True, capture_output=True)
    sys.stdout.write(train_proc.stdout)
    sys.stderr.write(train_proc.stderr)
    if train_proc.returncode != 0:
        raise SystemExit(train_proc.returncode)
    train_seconds = round(time.time() - t0, 2)

    fuse_cmd = [
        *fuse_cmd_prefix,
        "--model",
        args.hf_model,
        "--adapter-path",
        str(ADAPTER_DIR),
        "--save-path",
        str(FUSED_DIR),
    ]
    if "--dequantize" in fuse_help:
        fuse_cmd.append("--dequantize")
    elif "--de-quantize" in fuse_help:
        fuse_cmd.append("--de-quantize")

    print(f"[2/3] Fusing adapter into FP16 Safetensors: {' '.join(fuse_cmd)}", flush=True)
    t1 = time.time()
    fuse_proc = subprocess.run(fuse_cmd, check=True, text=True, capture_output=True)
    sys.stdout.write(fuse_proc.stdout)
    sys.stderr.write(fuse_proc.stderr)
    fuse_seconds = round(time.time() - t1, 2)

    gguf_out = ROOT / "models" / "shaanpatel-cv-pico-q4_k_m.gguf"
    print(f"[3/3] Exporting and quantizing Q4_K_M GGUF -> {gguf_out}", flush=True)
    t2 = time.time()
    subprocess.run([sys.executable, str(ROOT / "export_gguf.py"), "--out-gguf", str(gguf_out)], check=True)
    gguf_seconds = round(time.time() - t2, 2)

    val_losses: list[dict[str, float | int]] = []
    train_losses: list[dict[str, float | int]] = []
    combined = train_proc.stdout + "\n" + train_proc.stderr
    for raw_line in combined.splitlines():
        clean = re.sub(r"\x1b\[[0-9;]*m", "", raw_line).strip()
        m_val = re.match(r"^(\d+)\s+val\s+([0-9.]+)", clean)
        if m_val:
            val_losses.append({"iter": int(m_val.group(1)), "val_loss": float(m_val.group(2))})
            continue
        m_tr = re.match(r"^(\d+)\s+([0-9.]+)\s+[▼▲]", clean)
        if m_tr:
            train_losses.append({"iter": int(m_tr.group(1)), "train_loss": float(m_tr.group(2))})

    report = {
        "hf_base_model": args.hf_model,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "lora_scale": args.lora_scale,
        "num_layers": args.num_layers,
        "train_seconds": train_seconds,
        "fuse_seconds": fuse_seconds,
        "gguf_seconds": gguf_seconds,
        "fused_dir": str(FUSED_DIR),
        "quantized_gguf": str(gguf_out),
        "val_loss_curve": val_losses,
        "first_train_loss": train_losses[0] if train_losses else None,
        "last_train_loss": train_losses[-1] if train_losses else None,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved post-training report to {REPORT_PATH}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
