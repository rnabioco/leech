---
description: Run all linting steps and fix issues automatically
---

Run the following linting and formatting steps in order:

1. **Format code with ruff**: Run `uv run ruff format --exclude notebooks/ .` to auto-format all Python files (excluding notebooks)
2. **Fix linting issues**: Run `uv run ruff check --fix --exclude notebooks/ .` to automatically fix linting issues (excluding notebooks)
3. **Check remaining linting issues**: Run `uv run ruff check --exclude notebooks/ .` to check for any remaining issues that couldn't be auto-fixed (excluding notebooks)
4. **Lint notebooks**: Run `if [ -d "notebooks" ]; then uv run nbqa ruff notebooks/; else echo "notebooks/ directory not found"; fi` to check notebooks for linting issues (if notebooks/ directory exists)
5. **Format Snakemake files**: Run `if [ -d "pipeline/workflow" ]; then uv run snakefmt pipeline/workflow/; else echo "pipeline/workflow/ directory not found"; fi` to format Snakemake workflow files
6. **Type check with mypy**: Run `uv run mypy src/leech/` to check for type errors

If any step fails or reports issues:
- For formatting and auto-fixable linting: these should be automatically fixed
- For Snakemake formatting: snakefmt automatically formats files in place
- For non-fixable linting issues: report them clearly to the user
- For type errors: report them clearly with file locations and suggested fixes

After running all checks, provide a summary of:
- What was fixed automatically
- Any remaining issues that need manual attention
- Overall status (✓ all checks passed, or ✗ issues found)
