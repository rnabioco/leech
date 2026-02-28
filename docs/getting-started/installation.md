# Installation

This guide covers how to install `leech` and its dependencies.

## Requirements

- Python 3.10 or higher
- Linux or macOS (Windows may work but is not officially supported)
- 8GB+ RAM recommended for training
- CUDA-capable GPU recommended (but not required)

## Installation Methods

### Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast, reliable Python package manager that handles virtual environments and dependencies automatically.

#### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Clone and Install leech

```bash
# Clone the repository
git clone https://github.com/rnabioco/leech.git
cd leech

# Install dependencies (creates .venv automatically)
uv sync

# Install with dev dependencies
uv sync --all-extras
```

### Using pip

If you prefer using pip:

```bash
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

```bash
# Using uv
uv run leech --help

# Using pip (with activated venv)
leech --help
```

You should see the help message with available commands.

## GPU Support

leech uses PyTorch for training. GPU acceleration is automatically enabled if CUDA is available.

To check if GPU is available:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

If you need to install CUDA support, refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Development Installation

For development, install with all optional dependencies:

```bash
# Using uv
uv sync --all-extras

# Using pip
pip install -e ".[dev]"
```

This installs additional tools for:

- **Testing**: pytest, pytest-cov
- **Linting**: ruff (replaces black + flake8)
- **Type checking**: mypy

## Troubleshooting

### POD5 Installation Issues

If you encounter issues installing `pod5`, you may need to install system dependencies:

**Ubuntu/Debian:**
```bash
sudo apt-get install python3-dev libhdf5-dev
```

**macOS:**
```bash
brew install hdf5
```

### pysam Installation Issues

pysam requires certain system libraries:

**Ubuntu/Debian:**
```bash
sudo apt-get install libbz2-dev liblzma-dev libcurl4-openssl-dev
```

**macOS:**
```bash
brew install bzip2 xz curl
```

### Permission Errors

If you get permission errors during installation, avoid using `sudo`. Instead:

1. Use a virtual environment (uv handles this automatically)
2. Or install in user mode: `pip install --user -e .`

## Next Steps

- [Quick Start](quick-start.md): Run your first leech command
