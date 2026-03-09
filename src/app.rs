use anyhow::{Context, Result};

use crate::data::{Entry, Resume, Section, SectionId, resume};
use crate::persona::{Answer, AnswerEngine, QaConfig};

#[derive(Clone, Debug)]
pub struct ChatTurn {
    pub question: String,
    pub answer: Answer,
}

pub struct App {
    pub resume: Resume,
    qa_engine: AnswerEngine,
    selected_section: usize,
    selected_item: usize,
    status: String,
    small_terminal: bool,
    small_terminal_tip_dismissed: bool,
    question_mode: bool,
    chat_panel: bool,
    question_input: String,
    chat_turns: Vec<ChatTurn>,
}

impl App {
    pub fn new(initial_section: Option<SectionId>, qa_config: QaConfig) -> Self {
        let resume = resume();
        let qa_engine = AnswerEngine::new(qa_config, &resume);
        let selected_section = initial_section
            .and_then(|target| {
                resume
                    .sections
                    .iter()
                    .position(|section| section.id == target)
            })
            .unwrap_or(0);

        Self {
            resume,
            qa_engine,
            selected_section,
            selected_item: 0,
            status: String::from(
                "Ready. Use h/l to move sections, j/k to move entries, and / to ask questions.",
            ),
            small_terminal: false,
            small_terminal_tip_dismissed: false,
            question_mode: false,
            chat_panel: false,
            question_input: String::new(),
            chat_turns: Vec::new(),
        }
    }

    pub fn selected_section(&self) -> &Section {
        &self.resume.sections[self.selected_section]
    }

    pub fn selected_entry(&self) -> Option<&Entry> {
        self.selected_section().items.get(self.selected_item)
    }

    pub fn section_index(&self) -> usize {
        self.selected_section
    }

    pub fn item_index(&self) -> Option<usize> {
        self.selected_entry().map(|_| self.selected_item)
    }

    pub fn status(&self) -> &str {
        &self.status
    }

    pub fn is_question_mode(&self) -> bool {
        self.question_mode
    }

    pub fn question_input(&self) -> &str {
        &self.question_input
    }

    pub fn show_chat_panel(&self) -> bool {
        self.chat_panel && !self.chat_turns.is_empty()
    }

    pub fn chat_turns(&self) -> &[ChatTurn] {
        &self.chat_turns
    }

    pub fn set_status(&mut self, status: impl Into<String>) {
        self.status = status.into();
    }

    pub fn sync_viewport(&mut self, width: u16, height: u16) {
        let is_small = width < 90 || height < 24;

        if is_small && !self.small_terminal {
            self.small_terminal_tip_dismissed = false;
        }

        if !is_small {
            self.small_terminal_tip_dismissed = false;
        }

        self.small_terminal = is_small;
    }

    pub fn should_show_small_terminal_tip(&self) -> bool {
        self.small_terminal && !self.small_terminal_tip_dismissed
    }

    pub fn dismiss_small_terminal_tip(&mut self) {
        if self.small_terminal {
            self.small_terminal_tip_dismissed = true;
            self.status = String::from("Small-terminal tip dismissed.");
        }
    }

    pub fn enter_question_mode(&mut self) {
        self.question_mode = true;
        self.chat_panel = true;
        self.question_input.clear();
        self.status = String::from("Ask about work, worldview, essays, or interests.");
    }

    pub fn append_question_char(&mut self, ch: char) {
        if self.question_mode && !ch.is_control() {
            self.question_input.push(ch);
        }
    }

    pub fn backspace_question(&mut self) {
        if self.question_mode {
            self.question_input.pop();
        }
    }

    pub fn cancel_question_mode(&mut self) {
        self.question_mode = false;
        self.question_input.clear();
        self.status = String::from("Ask cancelled.");
    }

    pub fn submit_question(&mut self) -> Result<()> {
        let question = self.question_input.trim().to_string();
        if question.is_empty() {
            self.status = String::from("Question is empty.");
            return Ok(());
        }

        self.question_mode = false;
        self.status = String::from("Thinking...");
        let answer = self.qa_engine.answer(&question)?;
        self.chat_turns.push(ChatTurn {
            question,
            answer: answer.clone(),
        });
        if self.chat_turns.len() > 4 {
            self.chat_turns.remove(0);
        }

        self.chat_panel = true;
        self.question_input.clear();
        self.status = format!("Answered with {}.", answer.mode.label());
        Ok(())
    }

    pub fn toggle_chat_panel(&mut self) {
        if self.chat_turns.is_empty() {
            self.status = String::from("No answers yet. Press / to ask a question.");
            return;
        }

        self.chat_panel = !self.chat_panel;
        self.status = if self.chat_panel {
            String::from("Showing chat answers.")
        } else {
            String::from("Showing resume details.")
        };
    }

    pub fn next_section(&mut self) {
        self.selected_section = (self.selected_section + 1) % self.resume.sections.len();
        self.selected_item = 0;
        self.status = format!("Section: {}", self.selected_section().title);
    }

    pub fn previous_section(&mut self) {
        self.selected_section = if self.selected_section == 0 {
            self.resume.sections.len() - 1
        } else {
            self.selected_section - 1
        };
        self.selected_item = 0;
        self.status = format!("Section: {}", self.selected_section().title);
    }

    pub fn first_section(&mut self) {
        self.selected_section = 0;
        self.selected_item = 0;
        self.status = format!("Section: {}", self.selected_section().title);
    }

    pub fn last_section(&mut self) {
        self.selected_section = self.resume.sections.len() - 1;
        self.selected_item = 0;
        self.status = format!("Section: {}", self.selected_section().title);
    }

    pub fn next_item(&mut self) {
        let len = self.selected_section().items.len();
        if len == 0 {
            self.status = String::from("This section is informational only.");
            return;
        }

        self.selected_item = (self.selected_item + 1) % len;
        if let Some(entry) = self.selected_entry() {
            self.status = format!("Entry: {}", entry.title);
        }
    }

    pub fn previous_item(&mut self) {
        let len = self.selected_section().items.len();
        if len == 0 {
            self.status = String::from("This section is informational only.");
            return;
        }

        self.selected_item = if self.selected_item == 0 {
            len - 1
        } else {
            self.selected_item - 1
        };

        if let Some(entry) = self.selected_entry() {
            self.status = format!("Entry: {}", entry.title);
        }
    }

    pub fn open_selected(&mut self) -> Result<()> {
        if let Some(entry) = self.selected_entry()
            && let Some(url) = entry.url
        {
            open::that(url).with_context(|| format!("failed to open {url}"))?;
            self.status = format!("Opened {}", entry.title);
            return Ok(());
        }

        open::that(self.resume.website)
            .with_context(|| format!("failed to open {}", self.resume.website))?;
        self.status = String::from("Opened portfolio website.");
        Ok(())
    }
}
