use anyhow::{Context, Result};

use crate::data::{Entry, Resume, Section, SectionId, resume};

pub struct App {
    pub resume: Resume,
    selected_section: usize,
    selected_item: usize,
    status: String,
    small_terminal: bool,
    small_terminal_tip_dismissed: bool,
}

impl App {
    pub fn new(initial_section: Option<SectionId>) -> Self {
        let resume = resume();
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
            selected_section,
            selected_item: 0,
            status: String::from("Ready. Use h/l to move sections and j/k to move entries."),
            small_terminal: false,
            small_terminal_tip_dismissed: false,
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
