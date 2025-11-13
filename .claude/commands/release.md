---
description: Prepare and create a new release for leech
---

You are an expert Python release manager preparing a new version of the leech package. Follow this comprehensive release workflow:

## Prerequisites

**CRITICAL**: Before starting the release process, you MUST:

1. Run `/check` to ensure all linting, formatting, and type checks pass
2. Run `/docs` to ensure all documentation is up-to-date
3. Wait for both commands to complete successfully before proceeding

If either command reports issues, **STOP** and fix them before continuing with the release.

## Phase 1: Version Planning

Parse the user's version input:
- **Explicit version** (e.g., `0.2.0`, `1.0.0-rc.1`): Use as-is
- **Release type** (`major`, `minor`, or `patch`): Bump current version accordingly
- **Interactive**: If no input provided, read current version from `pyproject.toml` and prompt user for release type

**Important**: If current version has a pre-release suffix (e.g., `-alpha`, `-beta`, `-rc.1`), preserve the suffix when bumping.

Current version location: `pyproject.toml` line 3 (`version = "x.y.z"`)

## Phase 2: Planning File Cleanup

Search the project root for any Claude planning files (files with `_` in names that look like planning artifacts):
- Example patterns: `_plan_*.md`, `_notes_*.txt`, planning documents with underscores

If found, ask user what to do:
1. **Move to docs/guides/**: Relocate to documentation (rename to remove underscores)
2. **Delete**: Permanently remove the files
3. **Keep**: Leave them at project root

## Phase 3: Changelog Generation

1. **Gather commits**: Run `git log --oneline --no-decorate $(git describe --tags --abbrev=0 2>/dev/null || git rev-list --max-parents=0 HEAD)..HEAD` to get commits since last release
   - If no tags exist, get all commits from initial commit

2. **Categorize commits** by conventional commit prefixes or content analysis:
   - **Features**: `feat:`, `add:`, new functionality
   - **Fixes**: `fix:`, `bug:`, corrections
   - **Improvements**: `refactor:`, `perf:`, enhancements
   - **Documentation**: `docs:`, documentation updates
   - **Internal**: `chore:`, `test:`, `ci:`, internal changes

3. **Generate changelog entry** for `CHANGELOG.md`:
   ```markdown
   ## [X.Y.Z] - YYYY-MM-DD

   ### Features
   - Concise user-facing description

   ### Fixes
   - Bug fix descriptions

   ### Improvements
   - Enhancement descriptions

   ### Documentation
   - Doc update descriptions
   ```

4. **Prepend to CHANGELOG.md**: Insert the new entry at the top (after the header), keeping existing entries below

**Style guidelines**:
- Focus on user-facing changes (what users will notice)
- Be concise but informative
- Skip purely internal changes unless they affect users
- Use present tense ("Add feature" not "Added feature")

## Phase 4: Quality Assurance

1. **Verify checks passed**: Confirm /check and /docs completed successfully
2. **Build package**: Run `uv build` to create wheel and sdist
3. **Verify build artifacts**:
   - Check `dist/leech-X.Y.Z-py3-none-any.whl` exists
   - Check `dist/leech-X.Y.Z.tar.gz` exists
   - Report file sizes
4. **Display staged changes**: Run `git diff --cached --stat` and show what will be committed

## Phase 5: Version Update

1. **Update pyproject.toml**: Change version on line 3 from old to new version
2. **Verify no other version strings**: Search for hardcoded version strings in:
   - `src/leech/__init__.py` (if it exists)
   - `docs/*.md` files (for version references in docs)
   - Update any found version references

## Phase 6: User Confirmation

Display a summary and ask for confirmation:

```
📦 Release Summary
==================
Version: X.Y.Z
Build artifacts:
  - dist/leech-X.Y.Z-py3-none-any.whl (ABC KB)
  - dist/leech-X.Y.Z.tar.gz (XYZ KB)

Changes to commit:
  M pyproject.toml
  M CHANGELOG.md
  [any other modified files]

Ready to proceed with commit and tag?
```

Wait for user confirmation before proceeding.

## Phase 7: Release Finalization

1. **Stage all changes**: `git add pyproject.toml CHANGELOG.md [any other updated files]`
2. **Create commit**: `git commit -m "chore: release vX.Y.Z"`
3. **Create annotated tag**: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
4. **Display next steps**:
   ```
   ✅ Release vX.Y.Z prepared locally

   Next steps:
   1. Review the commit: git show
   2. Push to GitHub: git push origin main --follow-tags
   3. GitHub Actions will automatically create the release
   4. Optional: Publish to PyPI with `uv publish`
   ```

## Important Notes

- **Do not push automatically**: Let user review and push manually
- **Preserve git history**: Use proper commit messages and annotated tags
- **Verify builds**: Always build and check artifacts before committing
- **Documentation first**: /check and /docs MUST pass before release
- **Keep CHANGELOG**: Maintain full history, only prepend new entries

## Error Handling

If any step fails:
- **Checks fail**: Stop and report issues, user must fix before continuing
- **Build fails**: Report error, check dependencies and pyproject.toml
- **Git conflicts**: Report and ask user to resolve manually
- **Version already exists**: Check if tag exists, confirm with user if override needed

## Post-Release Checklist

After user pushes to GitHub, remind them to:
- [ ] Verify GitHub Actions workflow completes
- [ ] Check GitHub Releases page for new release
- [ ] Consider publishing to PyPI if package is public: `uv publish`
- [ ] Update any dependent projects or documentation
- [ ] Announce release (if applicable)
