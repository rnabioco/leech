---
description: Run all linting steps and fix issues automatically
---

Run the following linting and formatting steps in order:

**Use `uvx`, not `uv run`, for every tool below.** None of `ruff`, `ty`,
`snakefmt` or `nbqa` is installed in the project venv, so `uv run <tool>`
re-syncs the environment to pull them in — and that sync drops the compiled
`leech_core` extension, after which `tests/test_rust_version_pairing.py` fails
and the Rust fast paths silently stop being exercised. `uvx` runs each tool in
its own ephemeral environment and never touches `.venv`. (If you do sync by
accident, `bash rust/build.sh` puts the extension back.)

1. **Format code with ruff**: Run `uvx ruff format --exclude notebooks/ .` to auto-format all Python files (excluding notebooks)
2. **Fix linting issues**: Run `uvx ruff check --fix --exclude notebooks/ .` to automatically fix linting issues (excluding notebooks)
3. **Check remaining linting issues**: Run `uvx ruff check --exclude notebooks/ .` to check for any remaining issues that couldn't be auto-fixed (excluding notebooks)
4. **Lint notebooks**: Run `if [ -d "notebooks" ]; then uvx --with ruff nbqa ruff --fix notebooks/; else echo "notebooks/ directory not found"; fi` to auto-fix linting issues in notebooks (if notebooks/ directory exists)
5. **Format Snakemake files**: Run `if [ -d "pipeline/workflow" ]; then uvx snakefmt pipeline/workflow/; else echo "pipeline/workflow/ directory not found"; fi` to format Snakemake workflow files
6. **Type check with ty**: Run `uvx ty check src/leech/` to check for type errors

Note that `uvx` resolves the latest release of each tool, which can surface
diagnostics a pinned older version missed — that is usually a real finding, not
a false alarm.

If any step fails or reports issues:
- For formatting and auto-fixable linting: these should be automatically fixed
- For Snakemake formatting: snakefmt automatically formats files in place
- For non-fixable linting issues: report them clearly to the user
- For type errors: report them clearly with file locations and suggested fixes

After running all checks, provide a summary of:
- What was fixed automatically
- Any remaining issues that need manual attention
- Overall status (✓ all checks passed, or ✗ issues found)
