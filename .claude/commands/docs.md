---
description: Review and update all project documentation
---

Perform a thorough documentation audit and update for the leech project. Follow these steps systematically:

## 1. Code Structure Review

First, analyze the current codebase structure:
- List all Python modules in `src/leech/` and `src/leech/models/`
- Identify all CLI commands in `cli.py` and their current options
- Review the Snakemake workflow structure (`pipeline/workflow/Snakefile` and `pipeline/workflow/rules/*.smk`)
- Document any new modules, features, or architectural changes

## 2. API Documentation Review

For each Python module in `src/leech/`, check the corresponding API doc in `docs/api/`:
- Verify the API doc exists and matches the current module structure
- Check for missing functions, classes, or methods
- Verify function signatures, parameters, and return types are current
- Update examples if they reference outdated APIs
- Note any modules that exist in code but lack API documentation

## 3. README Review

Review `README.md` and compare with actual codebase:
- Verify installation instructions match current dependencies (`pyproject.toml`)
- Check CLI usage examples match current `cli.py` commands and options
- Verify feature descriptions match implemented functionality
- Update any outdated file paths or module references
- Ensure quick start examples are tested and work correctly

## 4. Documentation Site Review

Review the `docs/` directory structure:
- **Getting Started** (`docs/getting-started/`): Verify installation, quick-start, and CLI usage guides are current
- **Guides** (`docs/guides/`): Check that technical guides reflect current implementation
- **API Reference** (`docs/api/`): Ensure all modules are documented with current signatures
- **Architecture** (`docs/architecture.md`): Verify architectural descriptions match current code organization
- **Data Preparation** (`docs/data_preparation.md`): Check against actual data preparation code
- Check for broken internal links between documentation pages

## 5. Snakemake Pipeline Review

Review the Snakemake workflow:
- Check that `pipeline/workflow/Snakefile` references are current
- Verify all rules in `pipeline/workflow/rules/*.smk` match the current CLI interface
- Update any rule descriptions that reference outdated options or modules
- Check that config file examples match current expected structure

## 6. CLAUDE.md Review

Review `CLAUDE.md` against the current codebase:
- Verify module organization section matches actual file structure
- Check that referenced classes and functions exist with correct signatures
- Update any outdated implementation details
- Remove references to deleted modules (e.g., if data_prep.py was refactored)
- Add any new important modules or architectural patterns

## 7. Update Documentation

Based on your findings, update the following files:
- `README.md`: Fix any outdated information, broken examples, or missing features
- API docs in `docs/api/*.md`: Update signatures, add missing functions, fix examples
- Guide docs in `docs/getting-started/` and `docs/guides/`: Update steps, commands, and explanations
- `CLAUDE.md`: Reflect current architecture and remove obsolete references

## 8. Report Summary

After completing the review and updates, provide a summary report with:
- **Files Updated**: List all documentation files that were modified
- **Key Changes**: Highlight the most important corrections made
- **Missing Documentation**: Note any code that lacks documentation
- **Recommendations**: Suggest additional documentation improvements or gaps to fill

## Important Guidelines

- **Verify, don't assume**: Read the actual code files to verify current implementation
- **Test examples**: If you update CLI examples, verify the commands are valid
- **Preserve formatting**: Maintain existing markdown formatting styles and conventions
- **Be thorough but concise**: Update what needs updating, but don't rewrite entire sections unnecessarily
- **Cross-reference**: Ensure consistency between README, API docs, and guides
- **Consider the user**: Documentation should be clear for both new users and developers
