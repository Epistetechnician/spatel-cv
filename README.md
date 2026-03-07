# `spatel`

An installable terminal CV for Shaan Patel.

This project ships a polished Rust TUI that lets people browse experience, foundations, education, skills, links, and install commands directly from the terminal.

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

For TUI
```sh
spatel
```

Optional CLI entry points:

```sh
spatel --about --print # open the about section   
spatel --experience --print # open the experience section
spatel --links --print # open the links section
spatel --interests --print # open the interests section
spatel --install --print # open the install section
spatel --all --print # print the full CV
```

## Controls

- `h` / `l`: move between sections
- `j` / `k`: move between entries
- `enter` or `o`: open the current link in your browser
- `g` / `G`: jump to first or last section
- `x` or `esc`: dismiss the small-terminal tip when it appears
- `q`: quit

## Links

- X: `https://x.com/epistetechnic`
- LinkedIn: `https://www.linkedin.com/in/shaan-patel21/`
- GitHub: `https://github.com/Epistetechnician`
- Telegram: `@epistetechnician`

## Release flow

The repository includes:

- `.github/workflows/ci.yml` for format, clippy, and test checks
- `install.sh` for curl-based installs from GitHub Releases
- `.github/workflows/release.yml` for tagged release builds
- platform archives for macOS Intel, macOS Apple Silicon, and Linux x86_64

Tagging a release with `v*` triggers the build pipeline and uploads release archives that match the installer's expected naming convention.
