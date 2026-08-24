# Installation

This guide covers how to install `leech` and its dependencies.

## Requirements

- Python 3.12 or higher
- Linux or macOS (Windows may work but is not officially supported)
- 8GB+ RAM recommended for training
- CUDA-capable GPU recommended (but not required)

## Installation Methods

### From PyPI (Recommended)

`leech` is published on PyPI, along with `leech-core`, its compiled accelerator.

```bash title="Bash" linenums="1"
# With uv
uv add "leech[rust]"

# With pip
pip install "leech[rust]"
```

#### About the `rust` extra

The `rust` extra installs `leech-core`, which accelerates data preparation and
inference (POD5 I/O, signal refinement, chunk extraction). It is optional —
every accelerated path has a pure-Python fallback, so `pip install leech` gives
a fully working install, just a slower one.

Wheels are published for **manylinux x86_64 and aarch64**. They are stable-ABI
(`abi3`) wheels, so one wheel serves CPython 3.12 and every later 3.x.

On any other platform (macOS, Windows, musl/Alpine) pip falls back to building
`leech-core` from its sdist, which needs a Rust toolchain **and** network access
to github.com — the `escapepod-signal` dependency is fetched from git rather
than crates.io. If that is inconvenient, install plain `leech` instead.

To confirm which path you are on:

```bash title="Bash" linenums="1"
check-rust
```

### From source (development)

[uv](https://docs.astral.sh/uv/) is a fast, reliable Python package manager that handles virtual environments and dependencies automatically.

#### Install uv

```bash title="Bash" linenums="1"
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Clone and Install leech

```bash title="Bash" linenums="1"
# Clone the repository
git clone https://github.com/rnabioco/leech.git
cd leech

# Install dependencies (creates .venv automatically)
uv sync

# Install with dev dependencies
uv sync --all-extras

# Build the Rust extension from the workspace
bash rust/build.sh
```

!!! warning "Rebuild after `uv sync`"

    `uv sync` can restore a cached `leech_core` build over a current one. Rerun
    `bash rust/build.sh` after any sync, and use `check-rust` to confirm the
    versions match.

### Using pip from a checkout

```bash title="Bash" linenums="1"
# Clone the repository
git clone https://github.com/rnabioco/leech.git
cd leech

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install
pip install -e .
```

## Verify Installation

After installation, verify that leech is installed correctly:

```bash title="Bash" linenums="1"
# Using uv
uv run leech --help

# Using pip (with activated venv)
leech --help
```

You should see the help message with available commands.

Check whether the Rust accelerator is active and correctly paired:

```bash title="Bash" linenums="1"
check-rust
```

## GPU Support

leech uses PyTorch for training. GPU acceleration is automatically enabled if CUDA is available.

To check if GPU is available:

```python title="Python" linenums="1"
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

If you need to install CUDA support, refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Development Installation

For development, install with all optional dependencies:

```bash title="Bash" linenums="1"
# Using uv
uv sync --all-extras

# Using pip
pip install -e ".[dev]"
```

This installs additional tools for:

- **Testing**: pytest, pytest-cov
- **Linting**: ruff (replaces black + flake8)
- **Type checking**: ty

## Troubleshooting

### POD5 Support (escapepod)

leech reads POD5 signal through `escapepod`, a Rust-backed reader that installs
as a prebuilt wheel from PyPI. It is a required dependency, so a plain
`uv sync` is enough — the `pod5` extra is kept as a no-op alias for existing
scripts. escapepod does not depend on the `pod5` Python package or `libhdf5`, so
no HDF5 system libraries are required.

### pysam Installation Issues

pysam requires certain system libraries:

**Ubuntu/Debian:**
```bash title="Bash" linenums="1"
sudo apt-get install libbz2-dev liblzma-dev libcurl4-openssl-dev
```

**macOS:**
```bash title="Bash" linenums="1"
brew install bzip2 xz curl
```

### Permission Errors

If you get permission errors during installation, avoid using `sudo`. Instead:

1. Use a virtual environment (uv handles this automatically)
2. Or install in user mode: `pip install --user -e .`

## Next Steps

- [Quick Start](quick-start.md): Run your first leech command
