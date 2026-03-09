use std::{
    env, fs,
    path::PathBuf,
    process::{Command, Output},
};

use anyhow::{Context, Result, bail};

use crate::data::Resume;
#[cfg(test)]
use crate::data::resume;

pub const DEFAULT_PERSONA_MODEL: &str = "shaanpatel-cv-pico";
pub const DEFAULT_BASE_MODEL: &str = "qwen2.5:0.5b";
pub const DEFAULT_QUANTIZATION: &str = "q4_K_M";

#[derive(Clone, Debug)]
pub struct QaConfig {
    pub persona_model: String,
    pub base_model: String,
    pub quantization: String,
    pub offline_only: bool,
}

impl Default for QaConfig {
    fn default() -> Self {
        Self {
            persona_model: DEFAULT_PERSONA_MODEL.to_string(),
            base_model: DEFAULT_BASE_MODEL.to_string(),
            quantization: DEFAULT_QUANTIZATION.to_string(),
            offline_only: false,
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
}

impl AnswerMode {
    pub fn label(&self) -> &str {
        match self {
            Self::GroundedOnly => "grounded corpus",
            Self::BaseModel(_) => "base model",
            Self::PersonaModel(_) => "persona model",
        }
    }
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

    pub fn build_persona_model(&self) -> Result<PathBuf> {
        self.ensure_model_available(&self.config.base_model)?;

        let modelfile_path = env::temp_dir().join("spatel-shaan-pico.Modelfile");
        fs::write(&modelfile_path, self.render_modelfile())
            .with_context(|| format!("failed to write {}", modelfile_path.display()))?;

        let args = vec![
            "create".to_string(),
            self.config.persona_model.clone(),
            "-f".to_string(),
            modelfile_path.display().to_string(),
            "-q".to_string(),
            self.config.quantization.clone(),
        ];
        if let Err(error) = self.run_ollama(&args) {
            let message = format!("{error:#}");
            if message.contains("quantization is only supported for F16 and F32 models") {
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

        let retrieved = self.retrieve(trimmed, 4);
        if retrieved.is_empty() {
            return Ok(self.unknown_answer(trimmed));
        }

        if !self.config.offline_only {
            if self.model_exists(&self.config.persona_model) {
                if let Ok(body) =
                    self.generate_with_model(&self.config.persona_model, trimmed, &retrieved)
                {
                    return Ok(self.finalize_answer(
                        body,
                        AnswerMode::PersonaModel(self.config.persona_model.clone()),
                        &retrieved,
                    ));
                }
            }

            if self.model_exists(&self.config.base_model) {
                if let Ok(body) =
                    self.generate_with_model(&self.config.base_model, trimmed, &retrieved)
                {
                    return Ok(self.finalize_answer(
                        body,
                        AnswerMode::BaseModel(self.config.base_model.clone()),
                        &retrieved,
                    ));
                }
            }
        }

        Ok(self.finalize_answer(
            self.synthesize_offline_answer(trimmed, &retrieved),
            AnswerMode::GroundedOnly,
            &retrieved,
        ))
    }

    fn render_modelfile(&self) -> String {
        let mut lines = vec![
            format!("FROM {}", self.config.base_model),
            "PARAMETER temperature 0.2".to_string(),
            "PARAMETER top_p 0.9".to_string(),
            "PARAMETER repeat_penalty 1.1".to_string(),
            "PARAMETER num_ctx 8192".to_string(),
            format!(
                "SYSTEM {}",
                triple_quote(&render_system_prompt(&self.corpus))
            ),
        ];

        for example in few_shots() {
            lines.push(format!("MESSAGE user {}", triple_quote(example.question)));
            lines.push(format!(
                "MESSAGE assistant {}",
                triple_quote(example.answer)
            ));
        }

        lines.join("\n\n")
    }

    fn retrieve(&self, question: &str, limit: usize) -> Vec<RetrievedDocument> {
        let tokens = normalized_tokens(question);
        if tokens.is_empty() {
            return Vec::new();
        }

        let mut ranked: Vec<_> = self
            .corpus
            .iter()
            .cloned()
            .filter_map(|doc| {
                let haystack = format!(
                    "{} {} {} {}",
                    doc.title,
                    doc.kind,
                    doc.tags.join(" "),
                    doc.text
                )
                .to_ascii_lowercase();
                let mut score = 0usize;

                for token in &tokens {
                    if haystack.contains(token) {
                        score += 1;
                    }

                    if doc.tags.iter().any(|tag| tag.eq_ignore_ascii_case(token)) {
                        score += 2;
                    }

                    if doc.title.to_ascii_lowercase().contains(token) {
                        score += 2;
                    }
                }

                score += match doc.kind.as_str() {
                    "persona" | "writing" => 2,
                    "experience" => 1,
                    _ => 0,
                };

                if score == 0 {
                    None
                } else {
                    Some(RetrievedDocument { doc, score })
                }
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
        let tokens = normalized_tokens(question);
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

                sentence_scores.push((overlap + item.score, sentence));
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
            body: "I do not have a grounded answer for that in the local Shaan corpus yet. Ask about current work, Halo Labs, NPC Capital, Columbia, public goods, sacred economics, Dream DAO, or long-term technical interests.".to_string(),
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
                tags: entry.meta.iter().map(|tag| tag.to_string()).collect(),
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
            "I work at the boundary between technical systems, institutional design, and story. Right now I am building around confidential stablecoins, verifiable AI, trusted execution environments, partnerships, documentation, and developer onboarding. The thread I keep following is making complex environments feel legible, navigable, and alive.",
            &["identity", "focus", "halo", "verifiable-ai", "tees"],
        ),
        doc(
            "working-style",
            "Working Style",
            "persona",
            "I am comfortable moving between research, operations, partnerships, product communication, onboarding, and documentation. I value rigor, aesthetics, and systems that stay humane as they scale. I like connecting theory, implementation, and adoption instead of treating them as separate worlds.",
            &["working-style", "operator", "research", "docs", "adoption"],
        ),
        doc(
            "worldview-sacred-economics",
            "Sacred Economics and Coordination",
            "writing",
            "I keep returning to sacred economics because money shapes social reality. I want crypto to become a moral technology instead of only a financial one. I care about systems that reward stewardship, circulation, contribution, and maintenance without collapsing into surveillance or compulsory intimacy. Privacy and autonomy still matter, so the challenge is to build cypherpunk infrastructure that supports more caring and durable coordination.",
            &[
                "sacred-economics",
                "privacy",
                "coordination",
                "public-goods",
            ],
        ),
        doc(
            "dream-dao-lessons",
            "Dream DAO Lessons",
            "writing",
            "Dream DAO reinforced a simple thesis for me: people take responsibility in proportion to how real the responsibility is. When young builders have actual treasury power, they show up differently. It also taught me that onboarding is a justice question, governance fatigue arrives early, and civic imagination has to be paired with administrative stamina.",
            &["dream-dao", "governance", "onboarding", "youth", "civic"],
        ),
        doc(
            "gitcoin-public-goods",
            "Gitcoin and Public Goods Funding",
            "writing",
            "Gitcoin matters to me because it turned caring about public goods into allocative power. Quadratic funding widened what became visible, but it also exposed the limits of identity systems, attention dynamics, and episodic grant rounds. My view now is that public-goods funding needs a stack: quadratic funding for early signal, retroactive rewards for demonstrated value, delegated evaluation for specialized judgment, and streaming support for long-lived maintenance.",
            &["gitcoin", "public-goods", "funding", "retro-funding", "qf"],
        ),
        doc(
            "education-and-research",
            "Education and Research",
            "persona",
            "My academic base is economics and business analytics at Columbia University with an environmental concentration. My research covered growth and development questions in Kenya and relationships between pollution indicators and infant mortality in India. That background shaped how I think about systems, incentives, public impact, and coordination.",
            &["columbia", "economics", "analytics", "research"],
        ),
        doc(
            "halo-labs",
            "Halo Labs Work",
            "experience",
            "At Halo Labs I have been leading research around confidential stablecoins, verifiable AI, TEEs, zk-proofs, partnerships, documentation, and developer onboarding. The work sits at the intersection of privacy, payments, proof systems, and practical developer experience.",
            &["halo", "stablecoins", "verifiable-ai", "tees", "zk"],
        ),
        doc(
            "npc-capital",
            "NPC Capital Work",
            "experience",
            "At NPC Capital I worked as a full-stack developer and researcher inside a liquid crypto fund partnered with Polygon. I built investor tooling, established cross-chain data pipelines, and tested real-yield strategies with Hyperliquid validators. That experience sharpened my interest in analytics, market infrastructure, and execution systems.",
            &["npc", "polygon", "hyperliquid", "analytics", "trading"],
        ),
        doc(
            "ecosystem-and-public-goods",
            "Ecosystem and Public Goods Work",
            "experience",
            "Across Celo, Dream DAO, Eco DAO, and Solana Foundation, I worked on public-goods strategy, open-source education, governance tooling, content pipelines, and ecosystem coordination. I care about the unglamorous layer of systems work that helps communities move from ideas to durable practice.",
            &["celo", "eco-dao", "solana", "public-goods", "governance"],
        ),
        doc(
            "grounding-and-interests",
            "Grounding and Interests",
            "persona",
            "What keeps me grounded is permaculture, DIY engineering, running, biking, cooking, yoga, soccer, chess, Go, and Catan. My curiosity keeps compounding around privacy-focused blockchain architecture, stablecoins, DeFi, payment rails, blockchain analytics, agentic systems, trading infrastructure, zero-knowledge proving, and Rust-heavy protocol engineering.",
            &["interests", "permaculture", "agents", "defi", "rust", "zk"],
        ),
        doc(
            "long-term-positioning",
            "Long-Term Technical Direction",
            "persona",
            "The strongest long-term directions for me combine privacy, payments, AI assurance, execution systems, and platform architecture. I am especially drawn to protocol work where secure intelligence, market infrastructure, and operational clarity reinforce each other instead of living in separate stacks.",
            &["direction", "payments", "ai", "platform", "protocols"],
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
         Answer questions in first person as Shaan when the user is asking about his work, beliefs, preferences, or experiences.\n\
         Stay grounded in the corpus below. Do not invent employers, dates, credentials, projects, or personal history.\n\
         If the answer is not in the corpus, say that directly.\n\
         Keep answers concise, reflective, and specific.\n\n\
         PERSONAL CORPUS:\n",
    );

    for item in corpus {
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
         - keep the answer under 180 words\n\
         - use concrete details from the context\n\
         - if the context is insufficient, say so plainly\n\
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
            answer: "Right now I am focused on confidential stablecoins, verifiable AI, trusted execution environments, documentation, partnerships, and developer onboarding. I like work where privacy, proof systems, and clear operator experience reinforce each other.",
        },
        FewShot {
            question: "How do you think about public goods?",
            answer: "I care about turning care for shared infrastructure into durable funding and stewardship. Gitcoin, retro funding, and public-goods ecosystems matter to me because they make maintenance, education, and community work more legible instead of rewarding only spectacle.",
        },
        FewShot {
            question: "What did Dream DAO teach you?",
            answer: "Dream DAO taught me that people take responsibility in proportion to how real the responsibility is. It also showed me that onboarding is a justice question, governance fatigue is real, and civic imagination has to be backed by administrative stamina.",
        },
        FewShot {
            question: "How do you balance privacy with coordination?",
            answer: "I want more caring and durable forms of coordination without collapsing into surveillance. That is why I care about cypherpunk infrastructure, TEEs, and privacy-preserving systems that still make contribution and stewardship legible.",
        },
        FewShot {
            question: "What kind of work energizes you most?",
            answer: "I am most energized by work that makes high-complexity environments more legible. That usually means moving between research, systems design, docs, onboarding, and narrative clarity rather than staying in a single silo.",
        },
    ]
}

fn normalized_tokens(input: &str) -> Vec<String> {
    input
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .filter_map(|token| {
            let normalized = token.trim().to_ascii_lowercase();
            if normalized.len() < 3 || is_stopword(&normalized) {
                None
            } else {
                Some(normalized)
            }
        })
        .collect()
}

fn split_sentences(text: &str) -> Vec<String> {
    text.split(['.', '!', '?', '\n'])
        .map(str::trim)
        .filter(|sentence| sentence.len() > 24)
        .map(|sentence| {
            let mut line = sentence.to_string();
            if !line.ends_with('.') {
                line.push('.');
            }
            line
        })
        .collect()
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
        let mut config = QaConfig::default();
        config.offline_only = true;
        let engine = AnswerEngine::new(config, &resume());
        let answer = engine
            .answer("How do you think about public goods?")
            .expect("offline answer");

        assert!(answer.body.to_ascii_lowercase().contains("public goods"));
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
}
