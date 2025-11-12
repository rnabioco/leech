---
description: Run all linting steps and fix issues automatically
---

Run the following linting and formatting steps in order:

1. **Format code with ruff**: Run `uv run ruff format .` to auto-format all Python files
2. **Fix linting issues**: Run `uv run ruff check --fix .` to automatically fix linting issues
3. **Check remaining linting issues**: Run `uv run ruff check .` to check for any remaining issues that couldn't be auto-fixed
4. **Type check with mypy**: Run `uv run mypy src/leech/` to check for type errors

If any step fails or reports issues:
- For formatting and auto-fixable linting: these should be automatically fixed
- For non-fixable linting issues: report them clearly to the user
- For type errors: report them clearly with file locations and suggested fixes

After running all checks, provide a summary of:
- What was fixed automatically
- Any remaining issues that need manual attention
- Overall status (✓ all checks passed, or ✗ issues found)
