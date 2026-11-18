# Personal Model (`shaanpatel-cv-pico`)

`spatel` ships a post-trained, grounded personal Q&A stack for Shaan Patel covering his full operator record, essays, and six-month research monograph (through November 2026).

## Architecture

1. **Weight-Level SFT Post-Training (`posttrain/`)**:
   - `posttrain/build_dataset.py` deterministically builds ChatML training (`train.jsonl`, 138 examples across 36 domain groups), validation (`valid.jsonl`, 16 examples), raw ChatML (`train_chatml.txt`), and a zero-overlap held-out evaluation suite (`heldout_eval.jsonl`, 16 multi-fact and false-premise questions).
   - Examples train both **standalone parametric recall** (direct first-person questions) and **RAG synthesis** (`CONTEXT: / QUESTION: / ANSWER:` with distractors and epistemic scope boundaries).
   - `posttrain/run_posttrain.py` runs Apple Silicon Metal LoRA SFT (`rank=16, scale=2.0, warmup=20, cosine_decay, mask_prompt=true`) on `Qwen/Qwen2.5-0.5B-Instruct` via `mlx-lm` (`val_loss` `4.244 -> 0.049` across 260 iterations), fuses the trained adapter into FP16 Safetensors (`posttrain/fused_hf/`), and exports + quantizes `posttrain/models/shaanpatel-cv-pico-q4_k_m.gguf` (`379.4 MB`, `Q4_K_M`) via `posttrain/export_gguf.py` (`llama-quantize`).
2. **Ollama GGUF Packaging (`spatel --build-pico-model`)**:
   - Auto-discovers local post-trained GGUF weights (`--weights <PATH>`, `SPATEL_PICO_WEIGHTS`, or `posttrain/models/shaanpatel-cv-pico-q4_k_m.gguf`), falling back to `qwen2.5:0.5b` when building without local post-trained weights.
   - Registers `shaanpatel-cv-pico` in Ollama with the exact SFT ChatML template, stop tokens (`<|im_start|>`, `<|im_end|>`), and calibrated system prompt.
3. **IDF + Alias Grounded Retrieval (`src/persona.rs`)**:
   - Expands domain aliases (`wald` -> `holdout`, `kime`/`kailash` -> `kailas`, `toffoli`/`shor` -> `ecdsa`, `pca`/`olmo` -> `jspace`, `swe`/`gemma` -> `gemma4`, etc.), scores corpus documents with inverse document frequency (IDF) weights, and injects the top-3 grounded documents at query time.

## Held-Out Evaluation (`posttrain/eval_report.json`)

Evaluated on 16 strictly zero-overlap, echo-proof multi-clause and false-premise questions (`posttrain/data/heldout_eval.jsonl`, 32 total fact clauses):

| Runtime Mode | Strict Pass (Both Clauses) | Fact-Clause Recall |
| --- | --- | --- |
| `qwen2.5:0.5b` (`base_raw`, pre-posttraining) | `2 / 16` (`12.5%`) | `8 / 32` (`25.0%`) |
| `shaanpatel-cv-pico` (`pico_standalone`, `Q4_K_M` GGUF only) | `7 / 16` (`43.8%`) | `18 / 32` (`56.2%`) |
| `spatel --local-llm --ask` (`spatel_end_to_end`, post-trained GGUF + IDF RAG) | **`15 / 16` (`93.8%`)** | **`31 / 32` (`96.9%`)** |

## Commands

Ask a single grounded question (prefers local `shaanpatel-cv-pico` with `--local-llm`):

```sh
spatel --local-llm --ask "What have you built over the last six months?"
```

Interactive Q&A shell (`exit`, `quit`, `q`, or EOF to quit):

```sh
spatel --local-llm --chat
```

Reproduce weight post-training, Ollama registration, and held-out evaluation:

```sh
python3 -m venv posttrain/.venv
./posttrain/.venv/bin/pip install mlx-lm
./posttrain/.venv/bin/python posttrain/run_posttrain.py
spatel --build-pico-model
./posttrain/.venv/bin/python posttrain/eval_heldout.py
```

Pass an explicit Safetensors directory or GGUF file when building:

```sh
spatel --build-pico-model --weights ./posttrain/fused_hf
```

Force offline retrieval-only synthesis (no model runtime required):

```sh
spatel --ask "What confound did you catch in Temporal Grid Safety v16?" --offline-only
```

## Runtime Order

- With `--local-llm` (or when no `ANTHROPIC_API_KEY` / `MINIMAX_API_KEY` is set):
  1. `shaanpatel-cv-pico` in local Ollama
  2. `qwen2.5:0.5b` base model in local Ollama
  3. deterministic offline sentence synthesis from the embedded corpus
- When `ANTHROPIC_API_KEY` or `MINIMAX_API_KEY` is set in the environment and `--local-llm` is omitted, `spatel` tries the hosted Anthropic-compatible endpoint first unless `--no-remote-llm` is passed.

## Six-Month Corpus Coverage

- **Certified Agent Authority & Security**: `FormaSec` (Python/Kani-Rust/Lean 4 kernel equivalence across 600 cases, 0/552 hostile admissions), `Security Alignment OS` (279 Rust tests, 20/20 Intel TDX attested scenarios), `Lusis Mesh`, `Recoverable Ghost States`.
- **Adaptive Evaluation & Governed Evolution**: `Holdout Ledger` (0/59,200 certificate violations, 9/9 pre-registered hypotheses, Lean 4 proofs), `Playbox` (v0.3.2, Biscuit rooms, F3 H100 memorizer catch, 87.5% cross-model steering vs plain-English control), `Caduceus` (13 lanes, +0.245 Haiku / +0.305 Sonnet lift, 43/43 attacks refused, 41% tripwire false-hold kept disabled), `SWE-Gemma` (`gemma4-agent`, 2/34 E4B vs 9/34 31B) & `Devkit` (`runhygiene`, `trajhygiene`).
- **Mechanistic Interpretability, Continual Learning & Safety**: `Mech-Interp Unified` (Qwen3/Pythia agreement neurons + F2 0/8 static vs 3/8 probe planted-flaw audit), `J-Space Pilot` (0.00 DPO equivalence vs norm-matched PCA on OLMo-2/Tulu-3), `Temporal Grid Safety` (self-caught v16 learning-rate confound + v30 0.95 vs 0.27 discriminant safety adapters), `Theseus Lab` (GPT-2 permutation symmetry + V39 Jacobian derivative underfitting + 3.5x grader disagreement), `AHCL`, `HSI Lab`, and `cl7b` (`replay_kl` 0.997 retention).
- **Markets, Compute, ZK & Quantum**: `Statebook` (72-page whitepaper, `StateKeys`, 7 completeness reports, 3 finality clocks), `ECDSA.fail` (`arXiv:2609.09582`, -86.1% `Q x T` reversible `secp256k1` circuits), `HyperIntel`, `HyperAI`, `Mesh Brain`, `Orbital Mesh`, `Composed ZK Benchmark OS` (1,107 tests), `Aeneas` (~80 extraction phases, 102-command ledger), and `MeshQRC` (classical baseline win on real industrial data).
- **Computational Humanities, Essays & Operator Roles**: `kailas-corpus` (`2,626` blind-labeled passages in 15+ languages resolving a Spatial-Mandala vs. Multi-Temporal double dissociation across English, Sanskrit, and `282` recovered classical Tibetan `ཏི་སེ` & Bon Kangyur `གཡུང་དྲུང་དགུ་བརྩེགས` passages; `446` dual-coded pilgrimage windows including `368` public-domain episodes with Hajj/Holy Land `16.7%` vs Kailas `3.0%` calendar decoupling at `p = 0.0022`, and `48` locator-only Colin Thubron full-circuit kora episodes showing `20.0%` strict / `34.3%` loose route-aligned mandala phase along `r in [0, 1]`), 7 Commonplace Book humanities threads, all 17 essays, `Halo Labs` (Co-Founder & COO), `NPC Capital` (Polygon / Hyperliquid), `Solana`, `Celo`, `Dream DAO`, `Eco DAO`, `Columbia University`, spoken languages, and personal grounding.
