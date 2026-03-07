mod app;
mod data;
mod ui;

use std::{
    io::{self, IsTerminal},
    time::Duration,
};

use anyhow::Result;
use app::App;
use clap::{ArgGroup, Parser};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use data::{Entry, Resume, Section, SectionId};
use ratatui::{Terminal, backend::CrosstermBackend};

#[derive(Parser, Debug)]
#[command(
    name = "spatel",
    version,
    about = "Installable terminal CV for Shaan Patel",
    group(
        ArgGroup::new("section")
            .args([
                "about",
                "foundations",
                "experience",
                "education",
                "skills",
                "interests",
                "links",
                "install",
            ])
            .multiple(false)
    )
)]
struct Cli {
    #[arg(long, help = "Open on the overview section")]
    about: bool,
    #[arg(long, help = "Open on the foundations section")]
    foundations: bool,
    #[arg(long, help = "Open on the experience section")]
    experience: bool,
    #[arg(long, help = "Open on the education section")]
    education: bool,
    #[arg(long, help = "Open on the skills section")]
    skills: bool,
    #[arg(long, help = "Open on the interests and resources section")]
    interests: bool,
    #[arg(long, help = "Open on the links section")]
    links: bool,
    #[arg(long, help = "Open on the install section")]
    install: bool,
    #[arg(long, help = "Print the selected section and exit")]
    print: bool,
    #[arg(long, help = "Print the full CV and exit")]
    all: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let initial_section = cli.section();
    let app = App::new(initial_section);

    if cli.print || cli.all || !io::stdout().is_terminal() {
        if cli.all || initial_section.is_none() {
            print!("{}", format_resume(&app.resume));
        } else {
            print!("{}", format_section(app.selected_section()));
        }
        return Ok(());
    }

    run_tui(app)
}

impl Cli {
    fn section(&self) -> Option<SectionId> {
        if self.about {
            Some(SectionId::Overview)
        } else if self.foundations {
            Some(SectionId::Foundations)
        } else if self.experience {
            Some(SectionId::Experience)
        } else if self.education {
            Some(SectionId::Education)
        } else if self.skills {
            Some(SectionId::Skills)
        } else if self.interests {
            Some(SectionId::Interests)
        } else if self.links {
            Some(SectionId::Links)
        } else if self.install {
            Some(SectionId::Install)
        } else {
            None
        }
    }
}

fn run_tui(mut app: App) -> Result<()> {
    let mut terminal = TerminalSession::enter()?;
    run_event_loop(&mut terminal, &mut app)
}

fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    app: &mut App,
) -> Result<()> {
    loop {
        let area = terminal.size()?;
        app.sync_viewport(area.width, area.height);
        terminal.draw(|frame| ui::render(frame, app))?;

        if !event::poll(Duration::from_millis(200))? {
            continue;
        }

        match event::read()? {
            Event::Resize(width, height) => {
                app.sync_viewport(width, height);
            }
            Event::Key(key) => {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                if app.should_show_small_terminal_tip() {
                    match key.code {
                        KeyCode::Char('q') => return Ok(()),
                        KeyCode::Esc | KeyCode::Char('x') => {
                            app.dismiss_small_terminal_tip();
                        }
                        _ => {}
                    }
                    continue;
                }

                match key.code {
                    KeyCode::Char('q') | KeyCode::Esc => return Ok(()),
                    KeyCode::Char('h') | KeyCode::Left => app.previous_section(),
                    KeyCode::Char('l') | KeyCode::Right => app.next_section(),
                    KeyCode::Char('j') | KeyCode::Down => app.next_item(),
                    KeyCode::Char('k') | KeyCode::Up => app.previous_item(),
                    KeyCode::Char('g') => app.first_section(),
                    KeyCode::Char('G') => app.last_section(),
                    KeyCode::Char('x') => {}
                    KeyCode::Char('o') | KeyCode::Enter => {
                        if let Err(error) = app.open_selected() {
                            app.set_status(format!("Open failed: {error}"));
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

fn format_resume(resume: &Resume) -> String {
    let mut output = String::new();
    output.push_str(&format!("{}\n", resume.name.to_uppercase()));
    output.push_str(&format!("{}\n", resume.headline));
    output.push_str(&format!("{} | {}\n\n", resume.location, resume.website));

    for section in &resume.sections {
        output.push_str(&format_section(section));
        output.push('\n');
    }

    output
}

fn format_section(section: &Section) -> String {
    let mut output = String::new();
    output.push_str(&format!("{}\n", section.title.to_uppercase()));
    output.push_str(&format!("{}\n\n", section.description));

    for entry in &section.items {
        output.push_str(&format_entry(entry));
        output.push('\n');
    }

    output
}

fn format_entry(entry: &Entry) -> String {
    let mut output = String::new();
    output.push_str(&format!("{} | {}\n", entry.eyebrow, entry.title));
    output.push_str(&format!("{}\n", entry.subtitle));
    output.push_str(&format!("{}\n", entry.summary));

    if !entry.meta.is_empty() {
        output.push_str(&format!("tags: {}\n", entry.meta.join(", ")));
    }

    for bullet in entry.bullets {
        output.push_str(&format!("- {bullet}\n"));
    }

    if let Some(command) = entry.command {
        output.push_str(&format!("command: {command}\n"));
    }

    if let Some(url) = entry.url {
        output.push_str(&format!("link: {url}\n"));
    }

    output
}

struct TerminalSession {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
}

impl TerminalSession {
    fn enter() -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;
        Ok(Self { terminal })
    }
}

impl std::ops::Deref for TerminalSession {
    type Target = Terminal<CrosstermBackend<io::Stdout>>;

    fn deref(&self) -> &Self::Target {
        &self.terminal
    }
}

impl std::ops::DerefMut for TerminalSession {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.terminal
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formatted_resume_contains_core_sections() {
        let app = App::new(None);
        let output = format_resume(&app.resume);

        assert!(output.contains("OVERVIEW"));
        assert!(output.contains("EXPERIENCE"));
        assert!(output.contains("INTERESTS + RESOURCES"));
        assert!(output.contains("INSTALL"));
        assert!(output.contains("SHAAN PATEL"));
    }

    #[test]
    fn selected_section_prints_expected_heading() {
        let app = App::new(Some(SectionId::Links));
        let output = format_section(app.selected_section());

        assert!(output.starts_with("LINKS\n"));
        assert!(output.contains("Portfolio"));
    }

    #[test]
    fn small_terminal_tip_can_be_dismissed_and_resets() {
        let mut app = App::new(None);

        app.sync_viewport(80, 20);
        assert!(app.should_show_small_terminal_tip());

        app.dismiss_small_terminal_tip();
        assert!(!app.should_show_small_terminal_tip());

        app.sync_viewport(120, 40);
        assert!(!app.should_show_small_terminal_tip());

        app.sync_viewport(80, 20);
        assert!(app.should_show_small_terminal_tip());
    }
}
