# Personal Model

`spatel` now includes a grounded personal Q&A stack for Shaan Patel.

## What It Does

- builds a local Shaan knowledge corpus from the embedded CV data plus first-person worldview and essay summaries
- answers questions from that corpus even without a model runtime
- optionally builds and uses a small local Ollama persona model named `shaanpatel-cv-pico`
- exposes the flow through both CLI and TUI

## Commands

Single question:

```sh
spatel --ask "What are you working on right now?"
```

Interactive shell:

```sh
spatel --chat
```

The chat shell exits on `exit`, `quit`, `q`, or EOF.

Build the personalized local model:

```sh
spatel --build-pico-model
```

Force offline-only answers:

```sh
spatel --ask "How do you think about public goods?" --offline-only
```

## Runtime Modes

`spatel` answers in this order:

1. `shaanpatel-cv-pico` if it exists
2. the configured base Ollama model if it exists
3. offline retrieval-only synthesis from the local corpus

The default base model is `qwen2.5:0.5b`.

## Quantization Notes

The build command targets a small quantized model path, but Ollama only allows `-q` during `create` when the source weights are F16 or F32. `qwen2.5:0.5b` is already quantized, so the builder automatically retries without `-q` and keeps the small quantized base intact.

## Corpus Shape

The embedded corpus includes:

- resume sections and entries
- current focus and working-style notes
- Halo Labs, NPC Capital, and ecosystem experience summaries
- worldview notes around sacred economics, privacy, and coordination
- writing-grounded summaries for Dream DAO and Gitcoin/public-goods funding
- grounding interests and long-term technical direction

## TUI Flow

- press `/` or `?` to open the question prompt
- type the question and press `enter`
- press `tab` to switch between resume details and answer history

## Verification

Verified locally with:

```sh
cargo fmt
env PATH=/Users/shaanp/.rustup/toolchains/stable-aarch64-apple-darwin/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  RUSTC=/Users/shaanp/.rustup/toolchains/stable-aarch64-apple-darwin/bin/rustc \
  /Users/shaanp/.rustup/toolchains/stable-aarch64-apple-darwin/bin/cargo test
```

The local Ollama path was also verified by:

```sh
./target/debug/spatel --build-pico-model
./target/debug/spatel --ask "What are you working on right now?"
```
