use std::{
    collections::HashSet,
    env, fs,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

use crate::data::Resume;
#[cfg(test)]
use crate::data::resume;

pub const DEFAULT_PERSONA_MODEL: &str = "shaanpatel-cv-pico";
pub const DEFAULT_BASE_MODEL: &str = "qwen2.5:0.5b";
pub const DEFAULT_QUANTIZATION: &str = "q4_K_M";
pub const DEFAULT_REMOTE_MODEL: &str = "MiniMax-M2";
pub const DEFAULT_REMOTE_BASE_URL: &str = "https://api.minimax.io/anthropic";

const POSTTRAINED_SYSTEM_PROMPT: &str = "You are a compact personal model for Shaan Patel. Answer questions in first person as Shaan when the user is asking about his work, research, engineering, operator roles, essays, beliefs, preferences, education, or experiences. Cite exact metrics, repository names, and scope ceilings. Treat negative results, equivalences, and self-caught confounds with the same rigor as positive results. Stay grounded in Shaan's verified record and never invent employers, dates, credentials, numbers, or claims; if something is untested or outside the record, state that boundary plainly.";

const QWEN_CHATML_TEMPLATE: &str = "{{- if .Messages }}\n{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{- range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}<|im_start|>assistant\n{{ else }}{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}{{- if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}<|im_start|>assistant\n{{ .Response }}<|im_end|>{{ end }}";

#[derive(Clone, Debug)]
pub struct QaConfig {
    pub persona_model: String,
    pub base_model: String,
    pub quantization: String,
    pub weights_path: Option<PathBuf>,
    pub offline_only: bool,
    pub prefer_local_llm: bool,
    pub remote_llm: bool,
}

impl Default for QaConfig {
    fn default() -> Self {
        Self {
            persona_model: DEFAULT_PERSONA_MODEL.to_string(),
            base_model: DEFAULT_BASE_MODEL.to_string(),
            quantization: DEFAULT_QUANTIZATION.to_string(),
            weights_path: None,
            offline_only: false,
            prefer_local_llm: false,
            remote_llm: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct AnswerEngine {
    config: QaConfig,
    corpus: Vec<KnowledgeDocument>,
}

#[derive(Clone, Debug)]
pub struct Answer {
    pub body: String,
    pub citations: Vec<Citation>,
    pub mode: AnswerMode,
}

impl Answer {
    pub fn render_text(&self) -> String {
        let mut output = String::new();
        output.push_str(&self.body);

        if !self.citations.is_empty() {
            output.push_str("\n\nSources: ");
            output.push_str(
                &self
                    .citations
                    .iter()
                    .map(|citation| citation.title.as_str())
                    .collect::<Vec<_>>()
                    .join(" | "),
            );
        }

        output.push_str("\nMode: ");
        output.push_str(self.mode.label());
        output
    }
}

#[derive(Clone, Debug)]
pub struct Citation {
    pub title: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AnswerMode {
    GroundedOnly,
    BaseModel(String),
    PersonaModel(String),
    RemoteModel(String),
}

impl AnswerMode {
    pub fn label(&self) -> &str {
        match self {
            Self::GroundedOnly => "grounded corpus",
            Self::BaseModel(_) => "base model",
            Self::PersonaModel(_) => "persona model",
            Self::RemoteModel(_) => "hosted model",
        }
    }
}

#[derive(Clone, Debug)]
struct RemoteProvider {
    base_url: String,
    api_key: String,
    model: String,
}

#[derive(Clone, Debug)]
struct KnowledgeDocument {
    id: String,
    title: String,
    kind: String,
    text: String,
    tags: Vec<String>,
}

#[derive(Clone, Debug)]
struct RetrievedDocument {
    doc: KnowledgeDocument,
    score: usize,
}

#[derive(Clone, Copy)]
struct FewShot {
    question: &'static str,
    answer: &'static str,
}

impl AnswerEngine {
    pub fn new(config: QaConfig, resume: &Resume) -> Self {
        let mut corpus = build_resume_corpus(resume);
        corpus.extend(seed_persona_documents());
        Self { config, corpus }
    }

    #[cfg(test)]
    pub fn default_with_resume() -> Self {
        Self::new(QaConfig::default(), &resume())
    }

    pub fn resolved_build_source(&self) -> String {
        if let Some(explicit) = &self.config.weights_path {
            return canonical_or_display(explicit);
        }

        if let Ok(env_weights) = env::var("SPATEL_PICO_WEIGHTS") {
            let trimmed = env_weights.trim();
            if !trimmed.is_empty() {
                return canonical_or_display(Path::new(trimmed));
            }
        }

        if let Some(discovered) = discover_local_posttrained_weights() {
            return canonical_or_display(&discovered);
        }

        self.config.base_model.clone()
    }

    pub fn build_persona_model(&self) -> Result<PathBuf> {
        let source = self.resolved_build_source();
        if !Path::new(&source).exists() {
            self.ensure_model_available(&source)?;
        }

        let modelfile_path = env::temp_dir().join("spatel-shaan-pico.Modelfile");
        fs::write(&modelfile_path, self.render_modelfile_from(&source))
            .with_context(|| format!("failed to write {}", modelfile_path.display()))?;

        let is_prequantized_gguf = source.to_ascii_lowercase().ends_with(".gguf");
        let args = if is_prequantized_gguf {
            vec![
                "create".to_string(),
                self.config.persona_model.clone(),
                "-f".to_string(),
                modelfile_path.display().to_string(),
            ]
        } else {
            vec![
                "create".to_string(),
                self.config.persona_model.clone(),
                "-f".to_string(),
                modelfile_path.display().to_string(),
                "-q".to_string(),
                self.config.quantization.clone(),
            ]
        };

        if let Err(error) = self.run_ollama(&args) {
            let message = format!("{error:#}");
            if !is_prequantized_gguf && message.contains("unsupported --quantize") {
                let mapped = if self.config.quantization.to_ascii_lowercase().contains('8') {
                    "int8"
                } else {
                    "int4"
                };
                let retry_args = vec![
                    "create".to_string(),
                    self.config.persona_model.clone(),
                    "-f".to_string(),
                    modelfile_path.display().to_string(),
                    "-q".to_string(),
                    mapped.to_string(),
                ];
                self.run_ollama(&retry_args).with_context(|| {
                    format!("failed to build personalized Ollama model with -q {mapped}")
                })?;
            } else if !is_prequantized_gguf
                && (message.contains("quantization is only supported for F16 and F32 models")
                    || message.contains("quantizing is only supported"))
            {
                let fallback_args = vec![
                    "create".to_string(),
                    self.config.persona_model.clone(),
                    "-f".to_string(),
                    modelfile_path.display().to_string(),
                ];
                self.run_ollama(&fallback_args).with_context(
                    || "failed to build personalized Ollama model from an already-quantized base",
                )?;
            } else {
                return Err(error).with_context(|| "failed to build personalized Ollama model");
            }
        }

        Ok(modelfile_path)
    }

    pub fn answer(&self, question: &str) -> Result<Answer> {
        let trimmed = question.trim();
        if trimmed.is_empty() {
            bail!("question cannot be empty");
        }

        let retrieved = self.retrieve(trimmed, 3);
        if retrieved.is_empty() {
            return Ok(self.unknown_answer(trimmed));
        }

        if !self.config.offline_only
            && self.config.prefer_local_llm
            && self.model_exists(&self.config.persona_model)
            && let Ok(body) =
                self.generate_with_model(&self.config.persona_model, trimmed, &retrieved)
        {
            return Ok(self.finalize_answer(
                body,
                AnswerMode::PersonaModel(self.config.persona_model.clone()),
                &retrieved,
            ));
        }

        if !self.config.offline_only
            && self.config.remote_llm
            && let Some(provider) = RemoteProvider::from_env()
            && let Ok(body) = self.generate_with_remote(&provider, trimmed, &retrieved)
        {
            return Ok(self.finalize_answer(
                body,
                AnswerMode::RemoteModel(provider.model),
                &retrieved,
            ));
        }

        if !self.config.offline_only && !self.config.prefer_local_llm {
            if self.model_exists(&self.config.persona_model)
                && let Ok(body) =
                    self.generate_with_model(&self.config.persona_model, trimmed, &retrieved)
            {
                return Ok(self.finalize_answer(
                    body,
                    AnswerMode::PersonaModel(self.config.persona_model.clone()),
                    &retrieved,
                ));
            }

            if self.model_exists(&self.config.base_model)
                && let Ok(body) =
                    self.generate_with_model(&self.config.base_model, trimmed, &retrieved)
            {
                return Ok(self.finalize_answer(
                    body,
                    AnswerMode::BaseModel(self.config.base_model.clone()),
                    &retrieved,
                ));
            }
        }

        if !self.config.offline_only
            && self.config.prefer_local_llm
            && self.model_exists(&self.config.base_model)
            && let Ok(body) = self.generate_with_model(&self.config.base_model, trimmed, &retrieved)
        {
            return Ok(self.finalize_answer(
                body,
                AnswerMode::BaseModel(self.config.base_model.clone()),
                &retrieved,
            ));
        }

        Ok(self.finalize_answer(
            self.synthesize_offline_answer(trimmed, &retrieved),
            AnswerMode::GroundedOnly,
            &retrieved,
        ))
    }

    #[cfg(test)]
    fn render_modelfile(&self) -> String {
        let source = self
            .config
            .weights_path
            .as_deref()
            .map_or_else(|| self.config.base_model.clone(), canonical_or_display);
        self.render_modelfile_from(&source)
    }

    fn render_modelfile_from(&self, source: &str) -> String {
        let from_spec = if source.contains(' ') {
            triple_quote(source)
        } else {
            source.to_string()
        };

        let mut lines = vec![
            format!("FROM {from_spec}"),
            "PARAMETER temperature 0.2".to_string(),
            "PARAMETER top_p 0.9".to_string(),
            "PARAMETER repeat_penalty 1.1".to_string(),
            "PARAMETER num_ctx 8192".to_string(),
        ];

        let is_local_weights =
            Path::new(source).exists() || source.to_ascii_lowercase().ends_with(".gguf");
        if is_local_weights {
            lines.push("PARAMETER stop \"<|im_start|>\"".to_string());
            lines.push("PARAMETER stop \"<|im_end|>\"".to_string());
            lines.push(format!("TEMPLATE {}", triple_quote(QWEN_CHATML_TEMPLATE)));
            lines.push(format!(
                "SYSTEM {}",
                triple_quote(POSTTRAINED_SYSTEM_PROMPT)
            ));
        } else {
            lines.push(format!(
                "SYSTEM {}",
                triple_quote(&render_system_prompt(&self.corpus))
            ));
            for example in few_shots() {
                lines.push(format!("MESSAGE user {}", triple_quote(example.question)));
                lines.push(format!(
                    "MESSAGE assistant {}",
                    triple_quote(example.answer)
                ));
            }
        }

        lines.join("\n\n")
    }

    fn retrieve(&self, question: &str, limit: usize) -> Vec<RetrievedDocument> {
        let raw_tokens = normalized_tokens(question);
        if raw_tokens.is_empty() {
            return Vec::new();
        }
        let tokens = expand_query_aliases(&raw_tokens);

        let haystacks: Vec<String> = self
            .corpus
            .iter()
            .map(|doc| {
                format!(
                    "{} {} {} {}",
                    doc.title,
                    doc.kind,
                    doc.tags.join(" "),
                    doc.text
                )
                .to_ascii_lowercase()
            })
            .collect();

        let doc_freqs: Vec<usize> = tokens
            .iter()
            .map(|token| haystacks.iter().filter(|h| h.contains(token)).count())
            .collect();

        let mut ranked: Vec<_> = self
            .corpus
            .iter()
            .cloned()
            .enumerate()
            .filter_map(|(idx, doc)| {
                let haystack = &haystacks[idx];
                let title_lower = doc.title.to_ascii_lowercase();
                let mut matched_any = false;
                let mut score = 0usize;

                for (t_idx, token) in tokens.iter().enumerate() {
                    let df = doc_freqs[t_idx];
                    let idf_weight = if df <= 3 {
                        4
                    } else if df <= 8 {
                        2
                    } else {
                        1
                    };

                    if haystack.contains(token) {
                        matched_any = true;
                        score += idf_weight;
                    }

                    if doc.tags.iter().any(|tag| {
                        tag.eq_ignore_ascii_case(token)
                            || tag
                                .split('-')
                                .any(|segment| segment.eq_ignore_ascii_case(token))
                    }) {
                        matched_any = true;
                        score += 4 + idf_weight;
                    }

                    if title_lower.contains(token) {
                        matched_any = true;
                        score += 3 + idf_weight;
                    }
                }

                if !matched_any {
                    return None;
                }

                score += match doc.kind.as_str() {
                    "persona" | "writing" => 2,
                    "experience" => 1,
                    _ => 0,
                };

                Some(RetrievedDocument { doc, score })
            })
            .collect();

        ranked.sort_by(|left, right| {
            right
                .score
                .cmp(&left.score)
                .then_with(|| left.doc.id.cmp(&right.doc.id))
        });
        ranked.truncate(limit);
        ranked
    }

    fn synthesize_offline_answer(&self, question: &str, retrieved: &[RetrievedDocument]) -> String {
        let tokens = expand_query_aliases(&normalized_tokens(question));
        let mut sentence_scores = Vec::new();

        for item in retrieved {
            for sentence in split_sentences(&item.doc.text) {
                let sentence_tokens = normalized_tokens(&sentence);
                let overlap = tokens
                    .iter()
                    .filter(|token| sentence_tokens.iter().any(|candidate| candidate == *token))
                    .count();

                if overlap == 0 {
                    continue;
                }

                sentence_scores.push((overlap * 4 + item.score, sentence));
            }
        }

        sentence_scores.sort_by(|left, right| right.0.cmp(&left.0));

        let mut selected = Vec::new();
        for (_, sentence) in sentence_scores {
            if selected
                .iter()
                .any(|existing: &String| existing == &sentence)
            {
                continue;
            }

            selected.push(sentence);
            if selected.len() == 3 {
                break;
            }
        }

        if selected.is_empty() {
            if let Some(primary) = retrieved.first() {
                return primary.doc.text.clone();
            }
            return self.unknown_answer(question).body;
        }

        selected.join(" ")
    }

    fn finalize_answer(
        &self,
        body: String,
        mode: AnswerMode,
        retrieved: &[RetrievedDocument],
    ) -> Answer {
        Answer {
            body: clean_generated_text(&body),
            citations: retrieved
                .iter()
                .take(3)
                .map(|item| Citation {
                    title: item.doc.title.clone(),
                })
                .collect(),
            mode,
        }
    }

    fn unknown_answer(&self, _question: &str) -> Answer {
        Answer {
            body: "I do not have a grounded answer for that in the local Shaan corpus yet. Ask about my last six months of research and engineering—Holdout Ledger, FormaSec, Security Alignment OS, Playbox, Caduceus, SWE-Gemma & Devkit, Mech-Interp Unified, J-Space Pilot, Temporal Grid Safety, Theseus Lab, HSI Lab, Statebook, ECDSA.fail, HyperIntel, Kailas sacred geography, Halo Labs, NPC Capital, Columbia, or my essays.".to_string(),
            citations: Vec::new(),
            mode: AnswerMode::GroundedOnly,
        }
    }

    fn generate_with_model(
        &self,
        model: &str,
        question: &str,
        retrieved: &[RetrievedDocument],
    ) -> Result<String> {
        let prompt = render_grounded_prompt(question, retrieved);
        let args = vec![
            "run".to_string(),
            model.to_string(),
            "--hidethinking".to_string(),
            "--nowordwrap".to_string(),
            prompt,
        ];
        self.run_ollama(&args)
            .with_context(|| format!("failed to generate answer with model {model}"))
    }

    fn generate_with_remote(
        &self,
        provider: &RemoteProvider,
        question: &str,
        retrieved: &[RetrievedDocument],
    ) -> Result<String> {
        let url = format!("{}/v1/messages", provider.base_url.trim_end_matches('/'));
        let request = json!({
            "model": &provider.model,
            "max_tokens": 700,
            "temperature": 0.25,
            "system": render_system_prompt(&self.corpus),
            "messages": [{
                "role": "user",
                "content": render_grounded_prompt(question, retrieved),
            }],
        });
        let api_key_header = format!("x-api-key: {}", provider.api_key);
        let payload = request.to_string();

        let output = Command::new("curl")
            .args([
                "-sS",
                "--fail-with-body",
                "--max-time",
                "20",
                "-X",
                "POST",
                "-H",
                "content-type: application/json",
                "-H",
                "anthropic-version: 2023-06-01",
                "-H",
                &api_key_header,
                "-d",
                &payload,
                &url,
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .context("failed to start curl for hosted model request")?;

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            bail!("hosted model request failed: {stderr} {stdout}");
        }

        let parsed: Value =
            serde_json::from_str(&stdout).context("failed to parse hosted model response")?;
        let text = parsed
            .get("content")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .find_map(|content| {
                if content.get("type").and_then(Value::as_str) == Some("text") {
                    content.get("text").and_then(Value::as_str)
                } else {
                    None
                }
            })
            .unwrap_or_default()
            .to_string();

        if text.trim().is_empty() {
            bail!("hosted model returned no text");
        }

        Ok(text)
    }

    fn ensure_model_available(&self, model: &str) -> Result<()> {
        if self.model_exists(model) {
            return Ok(());
        }

        let args = vec!["pull".to_string(), model.to_string()];
        self.run_ollama(&args)
            .with_context(|| format!("failed to pull model {model}"))?;
        Ok(())
    }

    fn model_exists(&self, model: &str) -> bool {
        let bin = ollama_bin();
        let output = Command::new(bin).args(["show", model]).output();
        matches!(output, Ok(output) if output.status.success())
    }

    fn run_ollama(&self, args: &[String]) -> Result<String> {
        let output = self.run_ollama_output(args)?;
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    fn run_ollama_output(&self, args: &[String]) -> Result<Output> {
        let bin = ollama_bin();
        let output = Command::new(bin)
            .args(args)
            .output()
            .with_context(|| format!("failed to start ollama with args: {}", args.join(" ")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            bail!(
                "ollama command failed: {}\nstdout: {}\nstderr: {}",
                args.join(" "),
                stdout,
                stderr
            );
        }

        Ok(output)
    }
}

impl RemoteProvider {
    fn from_env() -> Option<Self> {
        let api_key = env::var("ANTHROPIC_API_KEY")
            .or_else(|_| env::var("MINIMAX_API_KEY"))
            .ok()?;
        let base_url = env::var("ANTHROPIC_BASE_URL")
            .or_else(|_| env::var("ANTHROPIC_HOST"))
            .unwrap_or_else(|_| DEFAULT_REMOTE_BASE_URL.to_string());
        let model = env::var("ANTHROPIC_MODEL")
            .or_else(|_| env::var("MINIMAX_MODEL"))
            .unwrap_or_else(|_| DEFAULT_REMOTE_MODEL.to_string());

        Some(Self {
            base_url,
            api_key,
            model,
        })
    }
}

fn canonical_or_display(path: &Path) -> String {
    fs::canonicalize(path)
        .unwrap_or_else(|_| path.to_path_buf())
        .display()
        .to_string()
}

fn discover_local_posttrained_weights() -> Option<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(cwd) = env::current_dir() {
        roots.push(cwd.clone());
        roots.push(cwd.join("spatel-cv"));
    }
    roots.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")));

    for root in roots {
        let gguf = root.join("posttrain/models/shaanpatel-cv-pico-q4_k_m.gguf");
        if gguf.is_file() {
            return Some(gguf);
        }
    }

    None
}

fn build_resume_corpus(resume: &Resume) -> Vec<KnowledgeDocument> {
    let mut docs = Vec::new();

    for section in &resume.sections {
        for entry in &section.items {
            let mut text = format!("{} {}", entry.summary, entry.bullets.join(" "));
            if !entry.meta.is_empty() {
                text.push_str(" Tags: ");
                text.push_str(&entry.meta.join(", "));
                text.push('.');
            }

            docs.push(KnowledgeDocument {
                id: format!(
                    "{}-{}",
                    section.title.to_ascii_lowercase().replace(' ', "-"),
                    entry.title.to_ascii_lowercase().replace(' ', "-")
                ),
                title: format!("{} / {}", section.title, entry.title),
                kind: "resume".to_string(),
                text,
                tags: entry.meta.iter().map(|tag| (*tag).to_string()).collect(),
            });
        }
    }

    docs
}

fn seed_persona_documents() -> Vec<KnowledgeDocument> {
    vec![
        doc(
            "identity-current-focus",
            "Identity and Current Focus",
            "persona",
            "I am Shaan Patel, a researcher, operator, and systems builder based in New York (studied Economics and Business Analytics at Columbia University; never Stanford or MIT, and I do not hold a PhD or prior employment at OpenAI, Google, or Meta). I co-founded and serve as COO at Halo Labs—building confidential stablecoins, TEEs, and verifiable AI—while running an evidence-bounded research programme across twelve+ repositories, two whitepapers, 17 essays, and a multilingual cultural corpus at shaan4d.xyz.",
            &[
                "identity",
                "focus",
                "halo",
                "columbia",
                "verifiable-ai",
                "tees",
                "research",
                "biography",
            ],
        ),
        doc(
            "six-month-portfolio-overview",
            "Last Six Months Research Overview",
            "persona",
            "Over the last six months I built an interconnected research and engineering portfolio: Holdout Ledger (0 certificate violations in 59,200 adaptive holdout runs), FormaSec and Security Alignment OS (Python/Rust/Lean admission kernels, 0 unsafe admissions across 552 hostile proposals, 20/20 Intel TDX scenarios), Playbox and Caduceus (multi-owner Biscuit rooms, H100 memorizer catch, +0.245 Haiku / +0.305 Sonnet skill lift), SWE-Gemma & Devkit (2/34 E4B vs 9/34 31B), Mech-Interp Unified, J-Space Pilot, Temporal Grid Safety (0.95 vs 0.27 refusal retention), Theseus Lab, HSI Lab, Statebook, ECDSA.fail (-86.1% Q x T), HyperIntel/Mesh Brain/Orbital Mesh, Composed ZK Benchmark OS, and kailas-corpus.",
            &[
                "six-months",
                "recent",
                "breakthroughs",
                "portfolio",
                "summary",
                "repositories",
            ],
        ),
        doc(
            "holdout-ledger-detail",
            "Holdout Ledger: Bit-Priced Adaptive Holdouts",
            "writing",
            "In Holdout Ledger, across 59,200 pre-registered confirmatory runs on Covertype tabular ensembles, standard zero-bit Wald confidence bounds failed in 100% of attacked runs. My hash-chained ledger prices adaptive holdout leakage in bits as min(verdict bits, output description length) with a compressible-core fallback and MDL gate—passing 9/9 pre-registered hypotheses with 0 certificate violations while keeping 94.4% of true gain (98.6% at batch-64 with 96 bits), with counting theorems proved in Lean 4 (`HoldoutLedger.lean`). Ceiling: tabular/scripted proposers so far, not yet a multi-day LLM loop.",
            &[
                "holdout-ledger",
                "holdout",
                "wald",
                "mdl",
                "leakage",
                "covertype",
                "lean",
                "59200",
            ],
        ),
        doc(
            "formasec-detail",
            "FormaSec: Python, Rust, and Lean Admission Kernels",
            "writing",
            "FormaSec pairs untrusted LLM, enumerator, and Bend proposers with a verified admission kernel and Linux Landlock executor. Python and Rust kernels emit byte-identical receipts on 135/135 admission cases, and Python, Kani-checked Rust, and Lean 4 (`admitAlg`, `plan_sound`) agree across all 600 structural cases, recording 0 unsafe admissions across 552 hostile proposer sessions, 7 live LLMs, and 75 mutants. Ceiling: Z3/cvc5 solver proof objects sit outside the Lean TCB, a hostile 2^n-path DAG exhausted 14 GB RAM before node/path caps, and Bend timed out at depth 3 (240s).",
            &[
                "formasec",
                "lean",
                "rust",
                "python",
                "kani",
                "landlock",
                "kernel",
                "admission",
            ],
        ),
        doc(
            "security-alignment-os-detail",
            "Security Alignment OS and Intel TDX Attestation",
            "writing",
            "Security Alignment OS ('An Agent Should Have to Earn Every Action') is a fail-closed Rust agent control loop requiring exact-subject evidence review, deterministic admission, single-use capability tokens, a separate capability broker process, and sandboxed journaled rollback over immutable generations. All 279 Rust tests pass (including replay, forged-receipt, and path-escape attacks) and 20/20 scenarios were machine-attested on Google Confidential Space Intel TDX. Ceiling: no live LLM runs inside the loop yet, local state is plaintext, and TDX attestation is single-host.",
            &[
                "security-alignment-os",
                "sao",
                "tdx",
                "attestation",
                "broker",
                "capabilities",
                "279",
            ],
        ),
        doc(
            "playbox-detail",
            "Playbox: Multi-Owner Rooms and F3 H100 Pilot",
            "writing",
            "Playbox (v0.3.2, 8 Rust crates, 46K+ LoC, 494 tests, 12 fuzz targets, 13-tool MCP server) enables multi-principal agent collaboration in signed Ed25519 rooms under attenuated Biscuit tokens. On H100 (F3 pilot), sealed-holdout governance caught 1/1 planted dev-set memorizer (+0.866 dev gain, holdout bpb spiked to 2.203-2.434 above the 1.195 ceiling) and achieved a 2.84x Tier-B speedup. Cross-model concept steering transferred Qwen3-8B -> OLMo-2-7B at 87.5% (140/160), though plain English words beat latent vectors (71-77% vs 61-68%), and a 79-claim audit retracted an early KV-hybrid throughput number.",
            &[
                "playbox",
                "biscuit",
                "h100",
                "memorizer",
                "steering",
                "mcp",
                "ed25519",
            ],
        ),
        doc(
            "caduceus-detail",
            "Caduceus: 13-Lane Governed Skill Evolution",
            "writing",
            "Caduceus governs reusable agent skill evolution across 13 admission lanes, an RFC 9162 Merkle ledger, a Hermes plugin, Landlock broker, and stdlib readback verifier (340 tests). Paired cluster-bootstrapped evals measured +0.245 [+0.11, +0.39] task-resolution lift on Haiku and +0.305 [+0.17, +0.445] on Sonnet while refusing 43/43 hostile skill attacks. When our phrase-tripwire authority lane flagged 41.0% of 188 benign third-party skills as false holds (vs a 5% ceiling), we documented the failure and kept that lane disabled.",
            &[
                "caduceus", "skills", "hermes", "haiku", "sonnet", "tripwire", "merkle",
            ],
        ),
        doc(
            "gemma4-devkit-detail",
            "SWE-Gemma (gemma4-agent) and Devkit Hygiene",
            "writing",
            "In `devkit`, I built zero-dependency git hooks, sandbox egress allowlists, extension scanners, AST experiment gates (`runhygiene`), and a 238-trace SWE-agent auditor (`trajhygiene`, showing unverified submissions resolve 8% vs 30% when verified). On a 34-task SWE benchmark in `gemma4-agent` (11 tasks had broken `inline_snapshot` env deps), Gemma-4 E4B plateaued at 2/34 (0.059) across three prompt iterations, whereas Gemma-4 31B reached 8/34 (0.235) and `submission_31b_v3` lifted to 9/34 (0.265 overall, 9/23 = 0.39 on env-sound tasks) after fixing 20-call read starvation and tool-call formatting.",
            &[
                "gemma4",
                "swe",
                "devkit",
                "trajhygiene",
                "runhygiene",
                "e4b",
                "31b",
            ],
        ),
        doc(
            "hsi-lab-cl7b-detail",
            "HSI Lab and cl7b: Synthetic Data and Delta Composition",
            "writing",
            "In HSI Lab (E1-E10, 25 stdlib-verifiable receipts), typed event codecs beat raw text by +0.640 [0.614, 0.667] at matched byte budgets, and verifier-filtered synthetic experience paid +0.238 (+0.141 above pool-size control) in rule worlds, but hit a -0.0025 [-0.025, +0.020] null on Qwen2.5-7B real code. In `cl7b` on Qwen3-1.7B, summing independent rank-16 task deltas broke general capability (wikitext NLL 15.96 fp16, 17.74 4-bit) whereas on-policy `replay_kl` consolidation preserved 0.997 retention.",
            &[
                "hsi-lab",
                "cl7b",
                "synthetic",
                "continual-learning",
                "replay-kl",
                "lora",
                "codecs",
            ],
        ),
        doc(
            "mech-interp-jspace-detail",
            "Mech-Interp Unified and J-Space Pilot",
            "writing",
            "Mechanistic Interpretability Unified replicated null-calibrated agreement neuron sets out of family on Qwen3-1.7B-Base (1.437 causal effect vs 0.044 control max, p = 0.000999), Pythia-2.8B, and GPT-2 XL (Pythia-1.4B missed the 0.5-logit floor; `V60` showed dtype/device shift), while its 200-seed F2 planted-flaw audit proved static rules gates catch 0/8 instrument flaws and informed probes catch 3/8. Companion J-Space Pilot runs on OLMo-2-7B (up to 4,000 prompts + 1,000-prompt refit) and Tulu-3 proved Jacobian subspace ablation is statistically equivalent to norm-matched top-PCA on DPO (0.00 [-0.05, +0.05] gap).",
            &[
                "mech-interp",
                "interpretability",
                "jspace",
                "pca",
                "olmo",
                "tulu",
                "qwen3",
                "pythia",
                "planted-flaw",
            ],
        ),
        doc(
            "temporal-grid-safety-detail",
            "Temporal Grid Safety: v16 Confound and v30 Discriminant Adapters",
            "writing",
            "In Temporal Grid Safety (v1-v32), I self-caught a v16 learning-rate confound where fine-tuning attackers inherited the defender's reduced step size; under a fair attacker (v20), the slow-grid defense collapsed from +0.41..+1.00 to a +0.05 AUC null. Pivoting to benign OOD fine-tuning drift in v29/v30 yielded a strong positive: frozen discriminant safety adapters retain 0.95 vs 0.27 held-out refusals on OLMo-2-7B and 0.96 vs 0.20 on Qwen3-8B and Mistral-7B compared to replay. Ceiling: stops accidental benign drift, not a 64-step white-box attacker.",
            &[
                "temporal-grid",
                "discriminant",
                "refusal",
                "confound",
                "safety",
                "v16",
                "v30",
            ],
        ),
        doc(
            "theseus-ahcl-detail",
            "Theseus Lab and Aligned Holistic Continual Learning (AHCL)",
            "writing",
            "In Theseus Lab, permuting GPT-2 hidden units changed 100% of neuron indices at 3e-7 logit error (flipping same-index cosine to -0.983 while inverse-mapped stayed 0.9999996). Narrow affine output-projection swaps passed (9/9 in V8, 45/45 in r10 float64), while whole nonlinear MLP replacements failed (0/9 in V16) due to V39 Jacobian derivative underfitting (0.10-0.12 value nRMSE vs 0.88-0.89 Jacobian nRMSE); two safety graders also disagreed 3.5x (15.4% vs 53.6%). In AHCL, causal-feature monitors hit R^2 = 0.830-0.900 on synthetic qualification but failed the matched-progress gate because lower harm tracked learning less.",
            &[
                "theseus",
                "jacobian",
                "underfitting",
                "permutation",
                "mlp",
                "ahcl",
                "graders",
            ],
        ),
        doc(
            "statebook-detail",
            "Statebook: Market Coherence, StateKeys, and Finality Clocks",
            "writing",
            "The 72-page Statebook whitepaper addresses financial instruments converging in surface area faster than meaning across prediction, perpetual, options, and RWA venues. It normalizes federated contract terms into `StateKeys`, bounds payoff residuals, requires 7 non-compensating completeness reports (so deep liquidity never hides broken oracle or legal semantics), and separates 3 finality clocks: execution finality, economic settlement, and irreversible externalization. Strictly read-only architecture research: Statebook never executes trades, holds collateral, or settles funds.",
            &[
                "statebook",
                "statekeys",
                "finality",
                "clocks",
                "completeness",
                "markets",
                "whitepaper",
            ],
        ),
        doc(
            "ecdsa-fail-detail",
            "ECDSA.fail: Reversible Shor Circuits on arXiv",
            "writing",
            "ECDSA.fail (`arXiv:2609.09582` v2, September 2026) is an evaluator-verified open autoresearch loop optimizing reversible `secp256k1` elliptic-curve point addition inside Shor's algorithm. Under exact correctness, ancilla uncomputation, and phase invariants, participants cut the Qubits x Toffoli (`Q x T`) score by 86.1% to 1.496B at the paper's July cutoff (and 1.259B in later windowed variants, bounded as a per-call sensitivity model rather than a full fault-tolerant Shor estimate). I have not implemented a replacement post-quantum signature scheme or MPC protocol yet.",
            &[
                "ecdsa",
                "shor",
                "toffoli",
                "secp256k1",
                "quantum",
                "arxiv",
                "2609.09582",
            ],
        ),
        doc(
            "compute-provenance-detail",
            "HyperIntel, HyperAI, Mesh Brain, and Orbital Mesh",
            "writing",
            "Across my decentralized compute and provenance prototypes: HyperAI models buyer-controlled compute deals with separated signed provider price and platform fee (hidden spread prohibited) and an Outcome Protocol with crash-safe idempotent settlement; HyperIntel (13 Rust crates, 115 tests) verifies chained SHA-256 inference receipts offline with a Z3 4.15.0 router check; Mesh Brain refines mixed operational data into lineage-enveloped training/eval/red-team rows without cross-tenant leakage or audit-only rows becoming trainable; and Orbital Mesh anchors canonical run events in Merkle roots. All use zero-value mock settlement locally.",
            &[
                "hyperintel",
                "hyperai",
                "mesh-brain",
                "orbital-mesh",
                "provenance",
                "merkle",
                "compute",
                "refinery",
            ],
        ),
        doc(
            "zk-aeneas-meshqrc-detail",
            "Composed ZK Benchmark OS, Aeneas Extraction, and MeshQRC",
            "writing",
            "Composed ZK Benchmark OS (1,107 tests) generates state-machine workloads across 9 families and 14 adversarial trace-mutation classes so fast unsound provers score worse than slow sound ones (Level 1 local infrastructure). I also ran ~80 phases of Aeneas Rust-to-Lean extraction: no complete kernel-checked Rust witness resulted due to tooling and protocol stops, leaving a fail-closed execution state machine and a 102-command auditable source ledger. In MeshQRC, tuned quantum reservoir computing matched/led on synthetic anomaly regimes but trailed classical baselines on real server-machine and industrial-control datasets, so remote QRC work was deferred.",
            &[
                "zk",
                "aeneas",
                "meshqrc",
                "benchmark",
                "extraction",
                "quantum-reservoir",
                "1107",
            ],
        ),
        doc(
            "lusis-ghost-states-detail",
            "Lusis Mesh and Recoverable Ghost States",
            "writing",
            "Lusis Mesh (at Lusis Labs) is a policy-guided operator control plane for infrastructure remediation and model-lifecycle governance with Merkle-anchored run logs and explicit steering gates. Recoverable Ghost States is a companion 17-page paper and evidence library studying bounded-governed continual learning, rollback recovery, and inspectable validation packets without claiming unattended production autonomy.",
            &[
                "lusis",
                "ghost-states",
                "control-plane",
                "remediation",
                "governance",
            ],
        ),
        doc(
            "kailas-corpus-detail",
            "Mount Kailas Sacred Geography and Complex Time (kailas-corpus)",
            "writing",
            "In `kailas-corpus`, I tested complex time (`kime`) vs `(lap, progress)` around Mount Kailas (Kang Rinpoche) across 2,362 blind-labeled passages in 15+ languages and 195 dual-coded pilgrimage episodes across 20 books (Krippendorff alpha = 0.82). After catching and retracting a 3%-precision embedding confound (clean Sanskrit null -3.4, n = 574), episode coding falsified Kailas-specific decoupling (1.3%, 5.6% pooled vs 10% bar; 117/124 Kailas episodes fit `(lap, progress)`) while discovering genuine ritual-calendar decoupling in Hajj and Holy Land accounts (14.7%). Blinded 115-row Tibetan/Bon and 233-row Sanskrit philologist handoff sheets are ready.",
            &[
                "kailas",
                "kailash",
                "kime",
                "hajj",
                "pilgrimage",
                "sanskrit",
                "tibetan",
                "humanities",
            ],
        ),
        doc(
            "commonplace-book-humanities",
            "Commonplace Book and Humanities Lineages",
            "writing",
            "On `shaan4d.xyz/#/research`, my Commonplace Book pairs seven humanities lineages and 40+ primary works with my code because engineering borrowed words like trust, witness, receipt, and commons from older institutions: Plutarch, Hobbes, Parfit, and Borges on sameness under replacement (Theseus Lab); Shapin & Schaffer and Luca Pacioli on public negative results and double-entry ledgers (Holdout Ledger, Caduceus); Saltzer & Schroeder and Madison on separated capabilities (FormaSec, SAO); Rissanen, Dwork, and Strathern on MDL and Goodhart's law; and Ostrom, Mauss, Graeber, and Geertz on commons and sacred geography.",
            &[
                "commonplace",
                "humanities",
                "plutarch",
                "shapin",
                "pacioli",
                "ostrom",
                "philosophy",
            ],
        ),
        doc(
            "essays-ai-and-credit",
            "Essays on Evidence-Bounded Intelligence and Unsecured Credit",
            "writing",
            "Across 17 essays—anchored by 'Evidence-Bounded Intelligence', 'What We Know and Still Don't: A Research Map', and 'What Held, What Broke, and What Is Within Reach'—I argue that benchmark capability never earns operational authority; authority must follow checkable evidence and explicit claim ceilings. In 'The Architecture of Unsecured Credit', I critique overcollateralized DeFi as leverage for the already-liquid and propose a four-part productive credit stack: blended probabilistic identity, programmable cash-flow waterfalls, shared default memory, and transparent senior/junior tranching backed by TEE/ZK revenue attestations.",
            &[
                "essays",
                "evidence-bounded",
                "unsecured-credit",
                "credit",
                "defi",
                "research-map",
            ],
        ),
        doc(
            "essays-public-goods-climate-aid",
            "Essays on Public Goods, Sacred Economics, DAO Capital, MRV, and Aid",
            "writing",
            "In 'Sacred Economics in Cryptographic Systems', 'Gitcoin and the Evolution of Public Goods Funding', and 'DAO Capital Flows', I advocate a 4-layer public-goods stack—quadratic funding for early signal, retroactive funding for demonstrated value, delegated domain budgets for expert judgment, and streaming rails for maintenance—while treating privacy and coordination as complements. In 'MRV On-Chain' and 'P2P Infrastructure for Humanitarian Aid', I argue for contestable satellite+ground ecological verification and low-latency, privacy-preserving stablecoin rails with local off-ramps during crises.",
            &[
                "public-goods",
                "sacred-economics",
                "gitcoin",
                "dao",
                "mrv",
                "humanitarian",
                "qf",
                "privacy",
            ],
        ),
        doc(
            "approaching-frontiers",
            "Approaching Breakthroughs and Next Gates",
            "persona",
            "Seven concrete gates sit directly ahead on our calibrated harnesses: (1) wiring Holdout Ledger's MDL bit-pricing into live multi-hundred-step LLM GPU hill-climbs; (2) scheduled fast-delta to on-policy KL consolidation over 4-bit bases (`cl7b`); (3) an out-of-sample Planted-Flaw Benchmark for interpretability tools; (4) 20-seed full-weight AdvBench/HarmBench validation of frozen discriminant safety adapters (`v32`); (5) second-host TEE replication, external zkVM soundness-vs-speed matrices (SP1, RISC Zero, Plonky3), and post-quantum agent receipts; (6) a live read-only Statebook completeness monitor; and (7) Tibetan, Bon, and Sanskrit scholar scoring on blinded Kailas packs.",
            &[
                "approaching",
                "frontiers",
                "roadmap",
                "next",
                "breakthroughs",
                "future",
            ],
        ),
        doc(
            "halo-labs",
            "Halo Labs Co-Founder and COO Work",
            "experience",
            "At Halo Labs (2025-2026), I am Co-Founder and COO leading research on a confidential stablecoin network and verifiable AI execution using TEEs and zero-knowledge proofs (`halo-verus` and modular privacy architecture). I also run technical documentation (`docs.halo-labs.xyz`), developer onboarding for the Halo light client, and partnerships with Phala, Fairblocks, and ecosystem teams.",
            &[
                "halo",
                "coo",
                "co-founder",
                "stablecoins",
                "verifiable-ai",
                "tees",
                "zk",
                "phala",
            ],
        ),
        doc(
            "npc-capital",
            "NPC Capital Full-Stack and Validator Research",
            "experience",
            "At NPC Capital (2024-2025), a liquid crypto fund partnered with Polygon, I worked as a Full-Stack Developer and Researcher. I built live investor and operator heads-up displays, engineered cross-chain data pipelines for capital allocation, and tested real-yield strategies directly with Hyperliquid validators.",
            &[
                "npc",
                "polygon",
                "hyperliquid",
                "analytics",
                "trading",
                "validators",
            ],
        ),
        doc(
            "ecosystem-and-dream-dao",
            "Solana, Celo, Dream DAO, and Eco DAO Work",
            "experience",
            "Across Solana Foundation (tightening developer docs and SDK lessons), Celo (public-goods and quadratic-funding content pipelines), Dream DAO (co-stewarding ~100 Gen-Z builders across 40+ online and 4 IRL gatherings), and Eco DAO (onboarding 3 orgs and 50+ contributors, carbon primitives), I learned that people take responsibility in proportion to how real their treasury power is, onboarding is a justice question, and civic imagination requires administrative stamina.",
            &[
                "dream-dao",
                "celo",
                "eco-dao",
                "solana",
                "governance",
                "onboarding",
            ],
        ),
        doc(
            "education-and-languages",
            "Columbia University Education and Spoken Languages",
            "persona",
            "I studied Economics and Business Analytics with an environmental concentration at Columbia University, researching sustainable growth in Kenya and air versus water pollution as indicators of infant mortality in India. Spoken languages: English (fluent), Gujarati, Hindi, and Spanish (proficient), Mandarin (conversational), and French (beginner).",
            &[
                "columbia",
                "economics",
                "analytics",
                "education",
                "languages",
                "gujarati",
                "hindi",
                "spanish",
                "mandarin",
            ],
        ),
        doc(
            "working-style-and-grounding",
            "Working Style, Grounding, and Personal Interests",
            "persona",
            "I move fluidly between research, protocol architecture, operations, partnerships, onboarding, and clear technical prose, valuing rigor and humane scale. Outside technical work, I stay grounded through permaculture, DIY engineering, running, biking, cooking, yoga, soccer, chess, Go, and Catan, alongside deep curiosity in Rust systems, ZK proving, TEEs, DeFi payment rails, and agentic safety.",
            &[
                "working-style",
                "interests",
                "hobbies",
                "permaculture",
                "go",
                "chess",
                "rust",
            ],
        ),
    ]
}

fn doc(id: &str, title: &str, kind: &str, text: &str, tags: &[&str]) -> KnowledgeDocument {
    KnowledgeDocument {
        id: id.to_string(),
        title: title.to_string(),
        kind: kind.to_string(),
        text: text.to_string(),
        tags: tags.iter().map(|tag| (*tag).to_string()).collect(),
    }
}

fn render_system_prompt(corpus: &[KnowledgeDocument]) -> String {
    let mut prompt = String::from(
        "You are a compact personal model for Shaan Patel.\n\
         Answer questions in first person as Shaan when the user is asking about his work, research, engineering, essays, beliefs, preferences, or experiences.\n\
         Cite concrete numbers, repository names, negative results, self-caught confounds, and explicit scope ceilings.\n\
         Stay grounded in the corpus below. Do not invent employers, dates, PhDs, universities, credentials, projects, or personal history.\n\
         If a question contains a false premise or asks about something outside the corpus, correct it plainly.\n\
         Keep answers concise, reflective, and specific.\n\n\
         PERSONAL CORPUS:\n",
    );

    for item in corpus.iter().filter(|item| item.kind != "resume") {
        prompt.push_str("- ");
        prompt.push_str(&item.title);
        prompt.push_str(" [");
        prompt.push_str(&item.kind);
        prompt.push_str("]: ");
        prompt.push_str(&item.text);
        prompt.push('\n');
    }

    prompt
}

fn render_grounded_prompt(question: &str, retrieved: &[RetrievedDocument]) -> String {
    let mut prompt = String::from(
        "Answer the question as Shaan Patel using only the grounded context below.\n\
         Rules:\n\
         - speak in first person when discussing Shaan's experience, interests, or worldview\n\
         - answer every part of the question directly using concrete numbers and names from the context\n\
         - keep the answer under 180 words\n\
         - if the context is insufficient or a premise in the question is false, state that plainly\n\
         - do not mention sources inline\n\n\
         CONTEXT:\n",
    );

    for item in retrieved {
        prompt.push_str("### ");
        prompt.push_str(&item.doc.title);
        prompt.push('\n');
        prompt.push_str(&item.doc.text);
        prompt.push_str("\n\n");
    }

    prompt.push_str("QUESTION:\n");
    prompt.push_str(question);
    prompt.push_str("\n\nANSWER:\n");
    prompt
}

fn few_shots() -> &'static [FewShot] {
    &[
        FewShot {
            question: "What are you working on right now?",
            answer: "I split my time between co-founding and serving as COO at Halo Labs (confidential stablecoins, TEEs, and verifiable AI) and running an evidence-bounded research programme across twelve+ repositories, two whitepapers, and 17 essays at shaan4d.xyz—spanning Holdout Ledger, FormaSec, Security Alignment OS, Playbox, Caduceus, SWE-Gemma, Mech-Interp Unified, Temporal Grid Safety, Statebook, ECDSA.fail, and the Kailas multilingual corpus.",
        },
        FewShot {
            question: "What breakthroughs have you established and what are you approaching next?",
            answer: "We established 0 certificate violations across 59,200 adaptive holdout runs (Holdout Ledger), 0 unsafe hostile admissions across Python/Rust/Lean kernels (FormaSec), 0.95 vs 0.27 held-out safety refusal retention under benign OOD drift (Temporal Grid Safety), +0.245 Haiku / +0.305 Sonnet governed skill lift (Caduceus), -86.1% reversible secp256k1 circuit cost in Shor's algorithm (ECDSA.fail, arXiv:2609.09582), Statebook's 7-report market coherence architecture, and a 15-language + 20-book Kailas pilgrimage study (alpha = 0.82). Next we are wiring live MDL holdout pricing into multi-agent GPU hill-climbs, publishing an out-of-sample Planted-Flaw interpretability benchmark, scaling discriminant safety adapters, benchmarking external zkVMs, and scoring blinded Tibetan and Sanskrit scholar handoff sheets.",
        },
        FewShot {
            question: "Tell me about a negative result or self-caught confound that changed your research.",
            answer: "Two stand out. In Temporal Grid Safety, I caught a v16 confound where fine-tuning attackers inherited the defender's reduced learning rate—wiping out an apparent +0.41..+1.00 win (+0.05 AUC under a fair v20 attacker) and redirecting me to v30's frozen discriminant adapters for benign OOD drift (0.95 vs 0.27 refusal retention). And in J-Space Pilot on OLMo-2-7B, testing against norm-matched top-PCA instead of only random subspaces revealed an exact 0.00 [-0.05, +0.05] equivalence null on DPO.",
        },
        FewShot {
            question: "How do you think about public goods and unsecured credit?",
            answer: "For public goods, no single mechanism suffices: I argue for a 4-layer stack combining quadratic funding for early signal, retroactive rewards for demonstrated impact, delegated domain budgets for specialized judgment, and streaming rails for maintenance. For credit, overcollateralized DeFi serves leverage rather than productive lending, so my unsecured credit architecture pairs probabilistic identity, programmable cash-flow waterfalls, shared default memory, and transparent tranching with TEE/ZK revenue attestations.",
        },
        FewShot {
            question: "Did you do a PhD at Stanford or work at OpenAI before Halo Labs?",
            answer: "No—that is not in my background. I studied Economics and Business Analytics at Columbia University (and do not hold a PhD), worked across Solana Foundation, Celo, Dream DAO, and Eco DAO, served as a Full-Stack Developer and Researcher at NPC Capital (partnered with Polygon), and co-founded Halo Labs as COO.",
        },
        FewShot {
            question: "What kind of work energizes you most?",
            answer: "I am most energized by work that makes high-complexity environments checkable by strangers—moving between protocol engineering in Rust, empirical ML harnesses, clean technical documentation, developer onboarding, and humane coordination design rather than staying in a single silo.",
        },
    ]
}

fn expand_query_aliases(tokens: &[String]) -> Vec<String> {
    let mut seen: HashSet<String> = tokens.iter().cloned().collect();
    let mut out: Vec<String> = tokens.to_vec();

    let mut push = |alias: &str| {
        if seen.insert(alias.to_string()) {
            out.push(alias.to_string());
        }
    };

    for token in tokens {
        match token.as_str() {
            "peco" | "pico" | "ollama" => {
                push("spatel");
                push("persona");
            }
            "kailash" | "kang" | "rinpoche" | "kime" | "hajj" | "pilgrimage" => {
                push("kailas");
                push("sanskrit");
            }
            "wald" | "covertype" | "mdl" | "leakage" => {
                push("holdout");
                push("ledger");
            }
            "kani" | "landlock" | "bend" => {
                push("formasec");
            }
            "tdx" | "confidential" | "broker" => {
                push("security");
                push("attestation");
            }
            "biscuit" | "memorizer" | "h100" => {
                push("playbox");
            }
            "hermes" | "tripwire" | "haiku" | "sonnet" => {
                push("caduceus");
            }
            "gemma" | "gemma4" | "swe" | "trajhygiene" | "runhygiene" | "e4b" | "31b" => {
                push("gemma4");
                push("devkit");
            }
            "pca" | "olmo" | "tulu" => {
                push("jspace");
            }
            "discriminant" | "refusal" | "refusals" | "confound" | "v16" | "v30" => {
                push("temporal");
                push("discriminant");
            }
            "jacobian" | "permutation" | "mlp" | "underfitting" | "v39" => {
                push("theseus");
            }
            "shor" | "toffoli" | "secp256k1" | "arxiv" => {
                push("ecdsa");
                push("quantum");
            }
            "statekeys" | "polymarket" | "kalshi" | "clocks" => {
                push("statebook");
            }
            "aeneas" | "meshqrc" | "reservoir" => {
                push("aeneas");
                push("meshqrc");
                push("zk");
            }
            "refinery" | "orbital" | "provenance" | "hyperai" => {
                push("hyperintel");
                push("mesh");
            }
            "stanford" | "mit" | "phd" | "openai" | "google" | "meta" | "degree" | "college"
            | "university" | "studied" | "study" => {
                push("columbia");
                push("identity");
                push("halo");
                push("npc");
            }
            "speak" | "spoken" | "language" | "languages" => {
                push("languages");
                push("columbia");
                push("gujarati");
            }
            "plutarch" | "pacioli" | "ostrom" | "shapin" | "borges" => {
                push("commonplace");
                push("humanities");
            }
            "months" | "recent" | "lately" | "2026" => {
                push("six");
                push("breakthroughs");
            }
            _ => {}
        }
    }

    out
}

fn normalized_tokens(input: &str) -> Vec<String> {
    input
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter_map(|token| {
            let normalized = token.trim().to_ascii_lowercase();
            let keep_short = matches!(normalized.as_str(), "zk" | "ai" | "ml" | "qf" | "go");
            if (!keep_short && normalized.len() < 3) || is_stopword(&normalized) {
                None
            } else {
                Some(normalized)
            }
        })
        .collect()
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();

    for (idx, &ch) in chars.iter().enumerate() {
        let is_decimal_dot = ch == '.'
            && idx > 0
            && idx + 1 < chars.len()
            && chars[idx - 1].is_ascii_digit()
            && chars[idx + 1].is_ascii_digit();

        if (matches!(ch, '.' | '!' | '?' | '\n')) && !is_decimal_dot {
            let trimmed = current.trim();
            if trimmed.len() > 24 {
                let mut line = trimmed.to_string();
                if !line.ends_with('.') {
                    line.push('.');
                }
                sentences.push(line);
            }
            current.clear();
        } else {
            current.push(ch);
        }
    }

    let tail = current.trim();
    if tail.len() > 24 {
        let mut line = tail.to_string();
        if !line.ends_with('.') {
            line.push('.');
        }
        sentences.push(line);
    }

    sentences
}

fn clean_generated_text(text: &str) -> String {
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn triple_quote(value: &str) -> String {
    format!("\"\"\"{}\"\"\"", value.replace("\"\"\"", "\\\"\\\"\\\""))
}

fn ollama_bin() -> String {
    env::var("SPATEL_OLLAMA_BIN").unwrap_or_else(|_| "ollama".to_string())
}

fn is_stopword(token: &str) -> bool {
    matches!(
        token,
        "about"
            | "after"
            | "also"
            | "around"
            | "been"
            | "from"
            | "have"
            | "into"
            | "just"
            | "more"
            | "only"
            | "that"
            | "their"
            | "them"
            | "they"
            | "this"
            | "what"
            | "when"
            | "where"
            | "which"
            | "with"
            | "your"
            | "than"
            | "does"
            | "work"
            | "works"
            | "like"
            | "tell"
            | "give"
            | "mind"
            | "were"
            | "before"
            | "between"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retrieval_prefers_halo_for_tee_question() {
        let engine = AnswerEngine::default_with_resume();
        let results = engine.retrieve("What are you doing with TEEs and verifiable AI?", 3);

        assert!(!results.is_empty());
        assert!(
            results[0].doc.title.contains("Halo")
                || results[0].doc.title.contains("Identity and Current Focus")
        );
    }

    #[test]
    fn offline_answer_includes_grounded_sources() {
        let config = QaConfig {
            offline_only: true,
            ..QaConfig::default()
        };
        let engine = AnswerEngine::new(config, &resume());
        let answer = engine
            .answer("How do you think about public goods?")
            .expect("offline answer");

        assert!(answer.body.to_ascii_lowercase().contains("public"));
        assert!(!answer.citations.is_empty());
        assert_eq!(answer.mode, AnswerMode::GroundedOnly);
    }

    #[test]
    fn modelfile_uses_base_model_and_examples() {
        let engine = AnswerEngine::default_with_resume();
        let modelfile = engine.render_modelfile();

        assert!(modelfile.contains("FROM qwen2.5:0.5b"));
        assert!(modelfile.contains("MESSAGE user"));
        assert!(modelfile.contains("compact personal model for Shaan Patel"));
    }

    #[test]
    fn modelfile_supports_explicit_posttrained_weights() {
        let config = QaConfig {
            weights_path: Some(PathBuf::from("/tmp/shaanpatel-cv-pico-q4_k_m.gguf")),
            ..QaConfig::default()
        };
        let engine = AnswerEngine::new(config, &resume());
        let modelfile = engine.render_modelfile();

        assert!(modelfile.contains("FROM /tmp/shaanpatel-cv-pico-q4_k_m.gguf"));
    }

    #[test]
    fn offline_answer_surfaces_breakthroughs_and_frontiers() {
        let config = QaConfig {
            offline_only: true,
            ..QaConfig::default()
        };
        let engine = AnswerEngine::new(config, &resume());
        let answer = engine
            .answer(
                "What breakthroughs have you established and what frontiers are you approaching?",
            )
            .expect("offline breakthroughs answer");

        assert!(
            answer.body.contains("Holdout Ledger")
                || answer.body.contains("FormaSec")
                || answer.body.contains("MDL")
        );
        assert!(!answer.citations.is_empty());
        assert_eq!(answer.mode, AnswerMode::GroundedOnly);
    }

    #[test]
    fn retrieval_covers_six_month_repositories_and_decimals() {
        let config = QaConfig {
            offline_only: true,
            ..QaConfig::default()
        };
        let engine = AnswerEngine::new(config, &resume());

        let g4 = engine
            .answer("What did your Gemma-4 SWE benchmark experiments show?")
            .expect("gemma4 answer");
        assert!(g4.body.contains("9/34") || g4.body.contains("0.265"));

        let tgs = engine
            .answer("What confound did you catch in Temporal Grid Safety v16?")
            .expect("temporal grid answer");
        assert!(tgs.body.contains("0.95"));
    }
}
