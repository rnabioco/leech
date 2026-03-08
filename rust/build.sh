#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
uv run maturin develop --release
echo "leech_core Rust extension built and installed"
