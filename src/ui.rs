use ratatui::{
    prelude::*,
    widgets::{Block, Clear, List, ListItem, ListState, Padding, Paragraph, Wrap},
};

use crate::{app::App, data::Entry};

pub fn render(frame: &mut Frame<'_>, app: &App) {
    frame.render_widget(Clear, frame.area());

    let root = Layout::vertical([
        Constraint::Length(5),
        Constraint::Min(18),
        Constraint::Length(3),
    ])
    .split(frame.area());

    render_header(frame, app, root[0]);
    render_body(frame, app, root[1]);
    render_footer(frame, app, root[2]);

    if app.should_show_small_terminal_tip() {
        render_small_terminal_notice(frame);
    }
}

fn render_header(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let section = app.selected_section();
    let text = Text::from(vec![
        Line::from(vec![
            Span::styled(app.resume.name, Style::new().fg(Color::Cyan).bold()),
            Span::raw("  "),
            Span::styled(
                section.title,
                Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(app.resume.headline),
        Line::from(format!(
            "{}  |  {}",
            app.resume.location, app.resume.website
        )),
    ]);

    let header = Paragraph::new(text)
        .block(
            Block::bordered()
                .title("spatel")
                .padding(Padding::horizontal(1)),
        )
        .wrap(Wrap { trim: true });

    frame.render_widget(header, area);
}

fn render_body(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let columns = Layout::horizontal([
        Constraint::Percentage(23),
        Constraint::Percentage(28),
        Constraint::Percentage(49),
    ])
    .split(area);

    render_sections(frame, app, columns[0]);
    render_entries(frame, app, columns[1]);
    render_details(frame, app, columns[2]);
}

fn render_sections(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let items: Vec<ListItem<'_>> = app
        .resume
        .sections
        .iter()
        .map(|section| {
            let style = if section.id == crate::data::SectionId::Interests {
                Style::new().fg(Color::Magenta).bold()
            } else {
                Style::new()
            };

            ListItem::new(Line::from(Span::styled(section.title, style)))
        })
        .collect();

    let list = List::new(items)
        .block(Block::bordered().title("Sections"))
        .highlight_style(Style::new().bg(Color::Blue).fg(Color::Black).bold())
        .highlight_symbol(">> ");

    let mut state = ListState::default().with_selected(Some(app.section_index()));
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_entries(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let section = app.selected_section();
    let items: Vec<ListItem<'_>> = if section.items.is_empty() {
        vec![ListItem::new(Line::from("No items in this section"))]
    } else {
        section
            .items
            .iter()
            .map(|entry| {
                ListItem::new(vec![
                    Line::from(Span::styled(
                        entry.eyebrow,
                        Style::new().fg(Color::DarkGray),
                    )),
                    Line::from(Span::styled(
                        entry.title,
                        Style::new().bold().fg(Color::White),
                    )),
                    Line::from(Span::styled(entry.subtitle, Style::new().fg(Color::Gray))),
                ])
            })
            .collect()
    };

    let list = List::new(items)
        .block(Block::bordered().title("Entries"))
        .highlight_style(Style::new().bg(Color::Green).fg(Color::Black).bold())
        .highlight_symbol(">> ")
        .repeat_highlight_symbol(true);

    let mut state = ListState::default().with_selected(app.item_index().or(Some(0)));
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_details(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let section = app.selected_section();
    let text = if let Some(entry) = app.selected_entry() {
        detail_text(section.description, entry)
    } else {
        Text::from(vec![
            Line::from(Span::styled(
                section.title,
                Style::new().fg(Color::Yellow).bold(),
            )),
            Line::from(""),
            Line::from(section.description),
        ])
    };

    let details = Paragraph::new(text)
        .block(
            Block::bordered()
                .title("Details")
                .padding(Padding::horizontal(1)),
        )
        .wrap(Wrap { trim: false });

    frame.render_widget(details, area);
}

fn render_footer(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let footer = Paragraph::new(Text::from(vec![
        Line::from(
            "keys: h/l sections  j/k entries  enter/o open link  g/G ends  x dismiss tip  q quit",
        ),
        Line::from(Span::styled(app.status(), Style::new().fg(Color::Cyan))),
    ]))
    .block(Block::bordered().title("Controls"))
    .wrap(Wrap { trim: true });

    frame.render_widget(footer, area);
}

fn render_small_terminal_notice(frame: &mut Frame<'_>) {
    let popup = centered_rect(60, 20, frame.area());
    let message = Paragraph::new("This TUI looks best in a terminal that is at least 90 columns wide and 24 rows tall.\n\nPress x or Esc to dismiss this tip.")
    .block(Block::bordered().title("Tip"))
    .wrap(Wrap { trim: true });

    frame.render_widget(Clear, popup);
    frame.render_widget(message, popup);
}

fn detail_text(section_description: &str, entry: &Entry) -> Text<'static> {
    let mut lines = vec![
        Line::from(Span::styled(
            entry.title,
            Style::new().fg(Color::Yellow).bold(),
        )),
        Line::from(Span::styled(entry.subtitle, Style::new().fg(Color::Gray))),
        Line::from(""),
        Line::from(section_description.to_string()),
        Line::from(""),
        Line::from(entry.summary.to_string()),
    ];

    if !entry.meta.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled("tags: ", Style::new().fg(Color::DarkGray)),
            Span::styled(entry.meta.join(" | "), Style::new().fg(Color::Magenta)),
        ]));
    }

    if !entry.bullets.is_empty() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "highlights",
            Style::new().fg(Color::Cyan).bold(),
        )));
        for bullet in entry.bullets {
            lines.push(Line::from(format!("• {bullet}")));
        }
    }

    if let Some(command) = entry.command {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "install command",
            Style::new().fg(Color::Cyan).bold(),
        )));
        lines.push(Line::from(command.to_string()));
    }

    if let Some(url) = entry.url {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "link",
            Style::new().fg(Color::Cyan).bold(),
        )));
        lines.push(Line::from(url.to_string()));
    }

    Text::from(lines)
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::vertical([
        Constraint::Percentage((100 - percent_y) / 2),
        Constraint::Percentage(percent_y),
        Constraint::Percentage((100 - percent_y) / 2),
    ])
    .split(area);

    Layout::horizontal([
        Constraint::Percentage((100 - percent_x) / 2),
        Constraint::Percentage(percent_x),
        Constraint::Percentage((100 - percent_x) / 2),
    ])
    .split(vertical[1])[1]
}
