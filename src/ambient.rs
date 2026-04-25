use std::time::{Duration, Instant};

use ratatui::{prelude::*, widgets::Paragraph};

use crate::data::SectionId;

const FRAME_INTERVAL: Duration = Duration::from_millis(180);

#[derive(Clone, Copy, Debug)]
pub enum Atmosphere {
    Clear,
    Night,
    Rain,
    Storm,
    Snow,
    Fog,
    Field,
}

#[derive(Clone, Debug)]
pub struct AmbientScene {
    tick: u64,
    last_tick: Instant,
}

impl AmbientScene {
    pub fn new() -> Self {
        Self {
            tick: 0,
            last_tick: Instant::now(),
        }
    }

    pub fn advance(&mut self) {
        if self.last_tick.elapsed() >= FRAME_INTERVAL {
            self.tick = self.tick.wrapping_add(1);
            self.last_tick = Instant::now();
        }
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }
}

pub fn atmosphere_for(section: SectionId) -> Atmosphere {
    match section {
        SectionId::Overview => Atmosphere::Clear,
        SectionId::Foundations => Atmosphere::Field,
        SectionId::Experience => Atmosphere::Storm,
        SectionId::Education => Atmosphere::Night,
        SectionId::Skills => Atmosphere::Rain,
        SectionId::Interests => Atmosphere::Snow,
        SectionId::Links => Atmosphere::Fog,
        SectionId::Install => Atmosphere::Clear,
    }
}

pub fn render(frame: &mut Frame<'_>, area: Rect, atmosphere: Atmosphere, tick: u64) {
    if area.width < 24 || area.height < 8 {
        return;
    }

    let text = Text::from(render_lines(area.width, area.height, atmosphere, tick));
    frame.render_widget(Paragraph::new(text), area);
}

fn render_lines(width: u16, height: u16, atmosphere: Atmosphere, tick: u64) -> Vec<Line<'static>> {
    let mut canvas = Canvas::new(width as usize, height as usize);

    match atmosphere {
        Atmosphere::Clear => {
            draw_sun(&mut canvas, tick);
            draw_clouds(&mut canvas, tick, Color::White);
            draw_signal_line(&mut canvas, tick, Color::Cyan);
        }
        Atmosphere::Night => {
            draw_stars(&mut canvas, tick);
            draw_moon(&mut canvas, tick);
            draw_signal_line(&mut canvas, tick, Color::Blue);
        }
        Atmosphere::Rain => {
            draw_clouds(&mut canvas, tick, Color::DarkGray);
            draw_rain(&mut canvas, tick, false);
            draw_signal_line(&mut canvas, tick, Color::Cyan);
        }
        Atmosphere::Storm => {
            draw_clouds(&mut canvas, tick, Color::DarkGray);
            draw_rain(&mut canvas, tick, true);
            draw_lightning(&mut canvas, tick);
        }
        Atmosphere::Snow => {
            draw_stars(&mut canvas, tick);
            draw_snow(&mut canvas, tick);
            draw_signal_line(&mut canvas, tick, Color::Magenta);
        }
        Atmosphere::Fog => {
            draw_fog(&mut canvas, tick);
            draw_signal_line(&mut canvas, tick, Color::Gray);
        }
        Atmosphere::Field => {
            draw_stars(&mut canvas, tick);
            draw_fireflies(&mut canvas, tick);
            draw_signal_line(&mut canvas, tick, Color::Green);
        }
    }

    draw_ground(&mut canvas);
    canvas.into_lines()
}

struct Cell {
    ch: char,
    color: Color,
}

struct Canvas {
    width: usize,
    height: usize,
    cells: Vec<Cell>,
}

impl Canvas {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: (0..width * height)
                .map(|_| Cell {
                    ch: ' ',
                    color: Color::Reset,
                })
                .collect(),
        }
    }

    fn put(&mut self, x: i32, y: i32, ch: char, color: Color) {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            return;
        }

        let index = y as usize * self.width + x as usize;
        self.cells[index] = Cell { ch, color };
    }

    fn text(&mut self, x: i32, y: i32, text: &str, color: Color) {
        for (offset, ch) in text.chars().enumerate() {
            self.put(x + offset as i32, y, ch, color);
        }
    }

    fn into_lines(self) -> Vec<Line<'static>> {
        self.cells
            .chunks(self.width)
            .map(|row| {
                Line::from(
                    row.iter()
                        .map(|cell| Span::styled(cell.ch.to_string(), Style::new().fg(cell.color)))
                        .collect::<Vec<_>>(),
                )
            })
            .collect()
    }
}

fn draw_clouds(canvas: &mut Canvas, tick: u64, color: Color) {
    let shapes = [
        ["   .--.   ", " .-(    ).", "(___.__)_)"],
        ["   _  _   ", " ( `   )_ ", "(    )   )"],
    ];
    let width = canvas.width as i32;
    for index in 0..3 {
        let shape = &shapes[index % shapes.len()];
        let y = 1 + (index as i32 * 3) % (canvas.height as i32 / 2).max(2);
        let span = width + 18;
        let x = ((tick as i32 / (index as i32 + 1)) + index as i32 * 23) % span - 14;
        for (line_index, line) in shape.iter().enumerate() {
            canvas.text(x, y + line_index as i32, line, color);
        }
    }
}

fn draw_sun(canvas: &mut Canvas, tick: u64) {
    let x = (canvas.width as i32 - 10).max(3);
    let y = 1;
    let ray = if tick % 2 == 0 { '+' } else { 'x' };
    canvas.put(x + 2, y, ray, Color::Yellow);
    canvas.text(x, y + 1, "\\ | /", Color::Yellow);
    canvas.text(x, y + 2, "- O -", Color::Yellow);
    canvas.text(x, y + 3, "/ | \\", Color::Yellow);
}

fn draw_stars(canvas: &mut Canvas, tick: u64) {
    let count = (canvas.width * canvas.height / 70).max(6);
    for index in 0..count {
        let x = ((index * 37 + tick as usize) % canvas.width) as i32;
        let y = ((index * 17 + tick as usize / 3) % (canvas.height / 2).max(1)) as i32;
        let ch = match (index as u64 + tick) % 3 {
            0 => '*',
            1 => '+',
            _ => '.',
        };
        canvas.put(x, y, ch, Color::White);
    }
}

fn draw_moon(canvas: &mut Canvas, tick: u64) {
    let x = (canvas.width as i32 - 8).max(1);
    let ch = if tick % 8 < 4 { 'C' } else { 'O' };
    canvas.put(x, 2, ch, Color::Gray);
}

fn draw_rain(canvas: &mut Canvas, tick: u64, storm: bool) {
    let step = if storm { 3 } else { 5 };
    for x in (0..canvas.width).step_by(step) {
        let y = ((x as u64 * 5 + tick * 2) % canvas.height as u64) as i32;
        let ch = if storm {
            if tick % 2 == 0 { '\\' } else { '/' }
        } else {
            '|'
        };
        canvas.put(
            x as i32,
            y,
            ch,
            if storm { Color::White } else { Color::Cyan },
        );
    }
}

fn draw_lightning(canvas: &mut Canvas, tick: u64) {
    if tick % 9 > 2 {
        return;
    }
    let x = (canvas.width / 2) as i32;
    for (dx, y) in [(0, 1), (-1, 2), (1, 3), (0, 4), (2, 5)] {
        canvas.put(x + dx, y, '/', Color::Yellow);
    }
}

fn draw_snow(canvas: &mut Canvas, tick: u64) {
    for index in 0..(canvas.width / 4).max(4) {
        let x = ((index * 11 + tick as usize / 2) % canvas.width) as i32;
        let y = ((index * 7 + tick as usize) % canvas.height) as i32;
        canvas.put(x, y, '*', Color::White);
    }
}

fn draw_fog(canvas: &mut Canvas, tick: u64) {
    for y in (1..canvas.height).step_by(3) {
        let offset = (tick as usize + y) % 12;
        let line = "~".repeat(canvas.width.saturating_sub(offset).min(64));
        canvas.text(offset as i32, y as i32, &line, Color::DarkGray);
    }
}

fn draw_fireflies(canvas: &mut Canvas, tick: u64) {
    let lower = canvas.height / 2;
    for index in 0..(canvas.width / 10).max(3) {
        let x = ((index * 19 + tick as usize) % canvas.width) as i32;
        let y = (lower + ((index * 5 + tick as usize / 2) % (canvas.height - lower).max(1))) as i32;
        let ch = if (index as u64 + tick) % 4 == 0 {
            '*'
        } else {
            '.'
        };
        canvas.put(x, y, ch, Color::Yellow);
    }
}

fn draw_signal_line(canvas: &mut Canvas, tick: u64, color: Color) {
    let y = (canvas.height as i32 / 2).max(1);
    for x in 0..canvas.width {
        if (x as u64 + tick) % 7 == 0 {
            canvas.put(x as i32, y, '.', color);
        }
    }
}

fn draw_ground(canvas: &mut Canvas) {
    let y = canvas.height as i32 - 1;
    for x in 0..canvas.width {
        let ch = if x % 5 == 0 { '^' } else { '_' };
        canvas.put(x as i32, y, ch, Color::Green);
    }
}
