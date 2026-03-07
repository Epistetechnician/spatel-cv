#!/bin/sh
set -eu

REPO="Epistetechnician/spatel-cv"
BINARY="spatel"

require_commands() {
    for cmd in curl mktemp sed tar uname; do
        command -v "$cmd" >/dev/null 2>&1 || {
            echo "Error: required command '$cmd' not found" >&2
            exit 1
        }
    done
}

detect_os() {
    case "$(uname -s)" in
        Darwin*) echo "macos" ;;
        Linux*) echo "linux" ;;
        *)
            echo "Error: unsupported operating system" >&2
            exit 1
            ;;
    esac
}

detect_arch() {
    case "$(uname -m)" in
        x86_64|amd64) echo "amd64" ;;
        arm64|aarch64)
            if [ "$(detect_os)" = "linux" ]; then
                echo "Error: Linux ARM releases are not configured yet" >&2
                exit 1
            fi
            echo "arm64"
            ;;
        *)
            echo "Error: unsupported architecture" >&2
            exit 1
            ;;
    esac
}

latest_tag() {
    resolved_url="$(curl -fsSL -I -o /dev/null -w '%{url_effective}' \
        "https://github.com/${REPO}/releases/latest")"
    tag="$(printf '%s' "$resolved_url" | sed 's#.*/##')"

    case "$resolved_url" in
        */releases|*/releases/)
            echo "Error: no GitHub release is published for ${REPO} yet." >&2
            echo "Try installing from source instead:" >&2
            echo "cargo install --git https://github.com/${REPO}.git --bin ${BINARY}" >&2
            exit 1
            ;;
    esac

    printf '%s\n' "$tag"
}

asset_name() {
    echo "${BINARY}-$(detect_os)-$(detect_arch).tar.gz"
}

download_url() {
    tag="$1"
    echo "https://github.com/${REPO}/releases/download/${tag}/$(asset_name)"
}

install_dir() {
    if [ "$(id -u)" -eq 0 ]; then
        echo "/usr/local/bin"
    else
        echo "$HOME/.local/bin"
    fi
}

main() {
    require_commands

    tag="$(latest_tag)"
    if [ -z "$tag" ] || [ "$tag" = "null" ] || [ "$tag" = "releases" ]; then
        echo "Error: could not resolve the latest release tag" >&2
        exit 1
    fi

    url="$(download_url "$tag")"
    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "$tmp_dir"' EXIT

    archive_path="${tmp_dir}/$(asset_name)"
    echo "Installing ${BINARY} from ${tag}"
    echo "Downloading $(asset_name)"

    curl -fL --retry 3 --retry-delay 1 "$url" -o "$archive_path"
    tar -xzf "$archive_path" -C "$tmp_dir"

    target_dir="$(install_dir)"
    mkdir -p "$target_dir"
    mv "${tmp_dir}/${BINARY}" "${target_dir}/${BINARY}"
    chmod +x "${target_dir}/${BINARY}"

    echo "Installed ${BINARY} to ${target_dir}/${BINARY}"

    case ":$PATH:" in
        *":${target_dir}:"*) ;;
        *)
            echo ""
            echo "Add this to your shell profile if needed:"
            echo "export PATH=\"\$PATH:${target_dir}\""
            ;;
    esac
}

main "$@"
