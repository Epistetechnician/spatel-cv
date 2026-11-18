#!/usr/bin/env python3
"""Evaluates strictly held-out questions (`heldout_eval.jsonl`) across:
1. Base `qwen2.5:0.5b` (raw Ollama, pre-posttraining baseline)
2. Post-trained `shaanpatel-cv-pico` (standalone Ollama weights + Modelfile, no external CLI RAG)
3. Full `spatel --ask --local-llm --no-remote-llm` (post-trained `shaanpatel-cv-pico` + upgraded corpus RAG)

Writes exact metrics and per-question outputs to `posttrain/eval_report.json`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
HELDOUT_PATH = ROOT / "data" / "heldout_eval.jsonl"
REPORT_PATH = ROOT / "eval_report.json"


def ask_ollama(ollama_bin: str, model: str, question: str) -> str:
    proc = subprocess.run(
        [ollama_bin, "run", model, "--hidethinking", "--nowordwrap", question],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return proc.stdout.strip()


def ask_spatel(spatel_bin: str, model: str, question: str) -> str:
    proc = subprocess.run(
        [spatel_bin, "--local-llm", "--no-remote-llm", "--model", model, "--ask", question],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return proc.stdout.strip()


def score_response(text: str, item: dict[str, object]) -> tuple[bool, bool, bool]:
    lower = text.lower()
    g1 = [str(x).lower() for x in item["must_contain_any"]]  # type: ignore[index]
    g2 = [str(x).lower() for x in item["must_also_contain_any"]]  # type: ignore[index]
    hit1 = any(tok in lower for tok in g1)
    hit2 = any(tok in lower for tok in g2)
    return hit1 and hit2, hit1, hit2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ollama-bin", default="/usr/local/bin/ollama")
    parser.add_argument("--base-model", default="qwen2.5:0.5b")
    parser.add_argument("--pico-model", default="shaanpatel-cv-pico")
    parser.add_argument("--spatel-bin", default=str(REPO_ROOT / "target" / "release" / "spatel"))
    args = parser.parse_args()

    items = [json.loads(line) for line in HELDOUT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

    modes = [
        ("base_raw", lambda q: ask_ollama(args.ollama_bin, args.base_model, q)),
        ("pico_standalone", lambda q: ask_ollama(args.ollama_bin, args.pico_model, q)),
        ("spatel_end_to_end", lambda q: ask_spatel(args.spatel_bin, args.pico_model, q)),
    ]

    summary: dict[str, object] = {"total_questions": len(items), "modes": {}, "details": []}
    per_question: list[dict[str, object]] = [
        {"id": it["id"], "question": it["question"], "results": {}} for it in items
    ]

    for mode_name, runner in modes:
        full_passes = 0
        part_hits = 0
        for idx, item in enumerate(items):
            q = str(item["question"])
            resp = runner(q)
            passed, h1, h2 = score_response(resp, item)
            if passed:
                full_passes += 1
            part_hits += int(h1) + int(h2)
            per_question[idx]["results"][mode_name] = {  # type: ignore[index]
                "passed": passed,
                "clause_1": h1,
                "clause_2": h2,
                "response": resp,
            }
        summary["modes"][mode_name] = {  # type: ignore[index]
            "strict_pass_count": full_passes,
            "strict_pass_rate": round(full_passes / len(items), 4),
            "fact_clause_recall": round(part_hits / (2 * len(items)), 4),
        }
        print(f"[{mode_name}] strict pass: {full_passes}/{len(items)} ({full_passes/len(items):.1%}), clause recall: {part_hits}/{2*len(items)} ({part_hits/(2*len(items)):.1%})", flush=True)

    summary["details"] = per_question
    REPORT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote held-out evaluation report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
