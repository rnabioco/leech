#!/usr/bin/env bash
# Build the leech_core Rust extension into the active environment.
# leech is now a mixed maturin project, so maturin runs from the repo root
# (where pyproject.toml with [tool.maturin] lives), not from rust/.
set -e
cd "$(dirname "$0")/.."
uv run maturin develop --release
echo "leech_core Rust extension built and installed"
