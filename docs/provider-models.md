# Provider Models

`spatel` answers from the embedded Shaan corpus first. Generation is a polishing layer over retrieved evidence, not an ungrounded chat mode.

## Default Remote Provider

When `ANTHROPIC_API_KEY` or `MINIMAX_API_KEY` is set, the default hosted provider is Minimax through its Anthropic-compatible API.

Environment variables:

```sh
export ANTHROPIC_BASE_URL="https://api.minimax.io/anthropic"
export ANTHROPIC_HOST="https://api.minimax.io/anthropic"
export ANTHROPIC_API_KEY="..."
export ANTHROPIC_MODEL="MiniMax-M2"
```

`ANTHROPIC_BASE_URL`, `ANTHROPIC_HOST`, and `ANTHROPIC_MODEL` are optional. The defaults are the Minimax Anthropic-compatible base URL and `MiniMax-M2`.

Never store API keys in the repository.

## Local Provider

Pass `--local-llm` to try Ollama before hosted generation:

```sh
spatel --local-llm --ask "How does Shaan think about public goods?"
```

The local path checks the personalized `shaanpatel-cv-pico` model first, then the configured base model.

## Offline Provider

Pass `--offline-only` to disable all model calls:

```sh
spatel --offline-only --ask "What is Shaan building?"
```

This mode uses deterministic retrieval and sentence synthesis from the local corpus.

## Precedence

Default precedence:

1. Hosted Anthropic-compatible provider when a key is present.
2. Local Ollama models when available.
3. Grounded offline synthesis.

With `--local-llm`:

1. Local Ollama models when available.
2. Hosted Anthropic-compatible provider when a key is present.
3. Grounded offline synthesis.
