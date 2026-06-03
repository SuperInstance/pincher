#!/usr/bin/env bash
# install.sh — One-command installer for PincherOS
# Usage: curl -fsSL https://raw.githubusercontent.com/SuperInstance/pincherOS/main/install.sh | bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
DIM='\033[2m'
RESET='\033[0m'

info()  { echo -e "  ${CYAN}→${RESET} $*"; }
ok()    { echo -e "  ${GREEN}✓${RESET} $*"; }
warn()  { echo -e "  ${YELLOW}!${RESET} $*"; }
fail()  { echo -e "  ${RED}✗${RESET} $*"; exit 1; }

echo ""
echo -e "  ${RED}${BOLD}PincherOS Installer${RESET}"
echo -e "  ${DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
echo ""

# --- Check Rust + cargo ---
info "Checking for Rust toolchain..."
if command -v cargo &>/dev/null && command -v rustc &>/dev/null; then
    RUST_VER=$(rustc --version 2>/dev/null || echo "unknown")
    ok "Found ${RUST_VER}"
else
    warn "Rust not found. Installing via rustup..."
    curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | sh -s -- -y
    # shellcheck source=/dev/null
    source "${HOME}/.cargo/env" 2>/dev/null || true
    if ! command -v cargo &>/dev/null; then
        fail "Rust installation failed. Please install manually: https://rustup.rs"
    fi
    ok "Rust installed successfully"
fi

# --- Determine repo directory ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/Cargo.toml" ] && [ -d "${SCRIPT_DIR}/pincher-core" ]; then
    REPO_DIR="${SCRIPT_DIR}"
else
    # Clone if not already in a repo
    info "Cloning PincherOS repository..."
    REPO_DIR="$(mktemp -d)/pincherOS"
    git clone https://github.com/SuperInstance/pincherOS.git "${REPO_DIR}"
    ok "Repository cloned"
fi

cd "${REPO_DIR}"

# --- Build release ---
info "Building release binary (this may take a few minutes)..."
if cargo build --release -p pincher-cli 2>&1; then
    ok "Build successful"
else
    fail "Build failed. Check the error output above."
fi

# --- Install binary ---
INSTALL_DIR="${HOME}/.local/bin"
mkdir -p "${INSTALL_DIR}"

BINARY_PATH="target/release/pincher"
if [ -f "${BINARY_PATH}" ]; then
    cp "${BINARY_PATH}" "${INSTALL_DIR}/pincher"
    chmod +x "${INSTALL_DIR}/pincher"
    ok "Binary installed to ${INSTALL_DIR}/pincher"
else
    fail "Built binary not found at ${BINARY_PATH}"
fi

# --- Ensure install dir is in PATH ---
if [[ ":${PATH}:" != *":${INSTALL_DIR}:"* ]]; then
    warn "${INSTALL_DIR} is not in your PATH"
    info "Adding to ~/.bashrc (and ~/.zshrc if present)..."
    echo 'export PATH="${HOME}/.local/bin:${PATH}"' >> "${HOME}/.bashrc"
    if [ -f "${HOME}/.zshrc" ]; then
        echo 'export PATH="${HOME}/.local/bin:${PATH}"' >> "${HOME}/.zshrc"
    fi
    export PATH="${INSTALL_DIR}:${PATH}"
    ok "PATH updated (restart your shell or source ~/.bashrc)"
fi

# --- Create data directory ---
DATA_DIR="${HOME}/.pincher"
mkdir -p "${DATA_DIR}"
ok "Data directory created at ${DATA_DIR}"

# --- Done ---
echo ""
echo -e "  ${GREEN}${BOLD}PincherOS installed successfully!${RESET}"
echo ""
echo -e "  ${DIM}Get started:${RESET}"
echo -e "  ${CYAN}pincher status${RESET}    — check engine status"
echo -e "  ${CYAN}pincher doctor${RESET}    — run health check"
echo -e "  ${CYAN}pincher teach${RESET}     — teach a new reflex"
echo -e "  ${CYAN}pincher do \"list files\"${RESET} — execute an intent"
echo ""
