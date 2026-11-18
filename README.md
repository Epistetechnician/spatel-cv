# `spatel`

An installable Rust terminal CV and post-trained personal pico model (`shaanpatel-cv-pico`) for Shaan Patel.

This project ships a fast terminal TUI (`ratatui`) for browsing Shaan's six-month research monograph, essays, experience, foundations, education, skills, and links, alongside a grounded personal Q&A stack:

- **Post-trained local pico model (`shaanpatel-cv-pico`)**: LoRA SFT post-trained on `Qwen2.5-0.5B-Instruct` across 36 domain groups covering all recent repositories, papers, negative results, self-caught confounds, essays, operator roles, and personal history, quantized to `q4_K_M` in Ollama
- **IDF + alias grounded retrieval**: deterministic local retrieval across 50+ structured CV and first-person research documents
- **Flexible runtime**: local Ollama (`--local-llm`), optional Anthropic-compatible hosted generation when `ANTHROPIC_API_KEY` is set, or zero-dependency `--offline-only` synthesis
- **Interactive TUI & CLI**: `/` or `?` inside the TUI, `spatel --chat`, or one-shot `spatel --ask "..."`

## Install

### Cargo

```sh
cargo install --git https://github.com/Epistetechnician/spatel-cv.git --bin spatel
```

### Latest release

```sh
curl -fsSL https://raw.githubusercontent.com/Epistetechnician/spatel-cv/master/install.sh | sh
```

### Manual GitHub release download

```sh
gh release download --repo Epistetechnician/spatel-cv --pattern "spatel-*"
```

### Build locally

```sh
git clone https://github.com/Epistetechnician/spatel-cv.git
cd spatel-cv
cargo install --path .
```

## Usage

Launch the interactive TUI:

```sh
spatel
```

CLI entry points:

```sh
spatel --about --print          # overview section
spatel --research --print       # research & breakthroughs monograph
spatel --experience --print     # operator & engineering roles
spatel --links --print          # portfolio & social handles
spatel --interests --print      # grounding & reading
spatel --install --print        # install commands
spatel --all --print            # print full CV
spatel --local-llm --ask "What have you built over the last six months?"
spatel --local-llm --chat       # interactive personal Q&A shell
spatel --build-pico-model       # build/update shaanpatel-cv-pico in Ollama (auto-detects post-trained weights)
spatel --build-pico-model --weights ./posttrain/fused_hf
spatel --no-remote-llm --ask "What did Holdout Ledger and FormaSec prove?"
spatel --offline-only --ask "How do you think about public goods?"
```

## Controls

- `h` / `l` or `Left` / `Right`: move between sections
- `j` / `k` or `Down` / `Up`: move between entries
- `/` or `?`: open the personal model prompt
- `tab`: toggle between resume details and Q&A history
- `enter` or `o`: open the current entry URL in your browser
- `g` / `G`: jump to first or last section
- `x` or `esc`: dismiss the small-terminal tip when it appears
- `q`: quit

## Post-Training & Personal Model

To post-train `shaanpatel-cv-pico` end-to-end on Apple Silicon (MLX) and register it in Ollama:

```sh
python3 -m venv posttrain/.venv
./posttrain/.venv/bin/pip install mlx-lm
./posttrain/.venv/bin/python posttrain/run_posttrain.py
spatel --build-pico-model
./posttrain/.venv/bin/python posttrain/eval_heldout.py
```

If local post-trained weights (`posttrain/fused_hf/` or `posttrain/models/*.gguf`) are not present, `spatel --build-pico-model` automatically pulls `qwen2.5:0.5b` and builds the persona model from the embedded corpus and few-shots.

Optional hosted Anthropic-compatible flow (e.g. MiniMax):

```sh
export ANTHROPIC_BASE_URL="https://api.minimax.io/anthropic"
export ANTHROPIC_API_KEY="..."
spatel --ask "What connects your AI safety work to Statebook?"
```

## Links

- Portfolio & Research Monograph: `https://shaan4d.xyz`
- GitHub: `https://github.com/Epistetechnician`
- X: `https://x.com/epistetechnic`
- LinkedIn: `https://www.linkedin.com/in/shaan-patel21/`
- Telegram: `@epistetechnician`

## Docs

- [`docs/personal-model.md`](docs/personal-model.md): post-training pipeline, corpus coverage, retrieval, and runtime modes
