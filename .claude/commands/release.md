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

Version locations (**all three must move together**):
- `pyproject.toml` `[project] version` — the `leech` package
- `pyproject.toml` `[project.optional-dependencies] rust` — the pin
  `leech-core==x.y.z`. This is what stops PyPI users pairing a current `leech`
  with a stale extension; `check_rust()` only *warns*.
- `rust/Cargo.toml` (`version = "x.y.z"`) — the `leech-core` extension.
  `rust/pyproject.toml` takes it from there via `dynamic = ["version"]`, so
  Cargo.toml is the only place to edit.

`tests/test_rust_version_pairing.py::TestDeclaredVersionsAgree` fails if any of
the three disagree, and the `check-version` job in `.github/workflows/release.yml`
fails the release if they disagree with the tag.

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

4. **Show changelog to user**: Display the generated changelog entry and inform user they can:
   - Edit CHANGELOG.md directly before confirming
   - Review and modify any entries
   - Add additional context or details

5. **Prepend to CHANGELOG.md**: Insert the new entry at the top (after the header), keeping existing entries below

**Style guidelines**:
- Focus on user-facing changes (what users will notice)
- Be concise but informative
- Skip purely internal changes unless they affect users
- Use present tense ("Add feature" not "Added feature")

**Note**: This changelog is for the project's CHANGELOG.md. GitHub release notes will be auto-generated from commits/PRs when you push the tag (configured in `.github/release.yml`). You can edit those on GitHub after the release is created.

## Phase 4: Quality Assurance

1. **Verify checks passed**: Confirm /check and /docs completed successfully
2. **Build package**: Run `uv build` to create wheel and sdist
3. **Verify build artifacts**:
   - Check `dist/leech-X.Y.Z-py3-none-any.whl` exists
   - Check `dist/leech-X.Y.Z.tar.gz` exists
   - Report file sizes
4. **Display staged changes**: Run `git diff --cached --stat` and show what will be committed

## Phase 5: Version Update

1. **Update pyproject.toml**: change `[project] version` **and** the
   `leech-core==` pin in the `rust` extra
2. **Update rust/Cargo.toml to the SAME version**: `leech-core` tracks
   `leech` exactly. This is not optional and not cosmetic:
   - `uv` keys its archive cache on this string, so a version that does not move
     lets `uv sync` restore a compiled extension built from *any* earlier
     revision that shared it — silently, over a current build.
   - `check_rust()` compares `leech_core.__version__` against
     `leech.__version__` and warns on a mismatch. That warning is only useful if
     the versions actually move together.

   It sat at `0.3.0` from v0.3.1 to v0.6.4 — ten releases — and did exactly the
   above.
3. **Rebuild and re-verify after the bump**: `bash rust/build.sh`, then confirm
   `python -c "from leech._rust_accel import check_rust; check_rust()"` prints
   the new version with no mismatch warning.
4. **Verify no other version strings**: Search for hardcoded version strings in:
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

1. **Stage all changes**: `git add pyproject.toml rust/Cargo.toml rust/Cargo.lock CHANGELOG.md uv.lock [any other updated files]`
2. **Create commit**: `git commit -m "chore: release vX.Y.Z"`
3. **Create annotated tag**: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
4. **Display next steps**:
   ```
   ✅ Release vX.Y.Z prepared locally

   Next steps:
   1. Review the commit: git show
   2. Push to GitHub: git push origin main --follow-tags
   3. GitHub Actions will automatically:
      - Verify the tag matches all three declared versions, then run the tests
      - Build leech-core abi3 wheels (manylinux x86_64 + aarch64) and sdists
      - Create a GitHub Release with every artifact attached
      - Mark releases with -alpha/-beta/-rc as pre-releases
      - Auto-generate release notes from commits/PRs
      - Include your CHANGELOG.md entry in the release
      - Publish BOTH `leech` and `leech-core` to PyPI
   4. Edit release notes on GitHub if needed (optional)
   ```

## Important Notes

- **Do not push automatically**: Let user review and push manually
- **Preserve git history**: Use proper commit messages and annotated tags
- **Verify builds**: Always build and check artifacts before committing
- **Documentation first**: /check and /docs MUST pass before release
- **Keep CHANGELOG**: Maintain full history, only prepend new entries

## PyPI publishing (one-time setup)

Releases publish to PyPI via **Trusted Publishing** (OIDC) — there is no API
token and no repository secret. If a release fails at the `publish-pypi-*` job
with an OIDC/permission error, the publisher is probably not registered.

Both projects need a publisher at <https://pypi.org/manage/account/publishing/>
(use *pending publisher* if the project does not exist yet):

| field | `leech` | `leech-core` |
|---|---|---|
| PyPI project name | `leech` | `leech-core` |
| Owner | `rnabioco` | `rnabioco` |
| Repository name | `leech` | `leech` |
| Workflow name | `release.yml` | `release.yml` |
| Environment name | `pypi-leech` | `pypi-leech-core` |

**The environment names must differ.** PyPI identifies a publisher by
`(owner, repo, workflow, environment)` and enforces a unique constraint on
exactly that tuple — the project name is *not* part of it. Two packages
released from one workflow under one environment name collide, and registering
the second fails with:

> A pending trusted publisher matching this configuration has already been
> registered for a different project name. Please contact PyPI's admins if this
> wasn't intentional.

That message is misleading here: nothing is wrong, no admin is needed, and it
does not mean someone took the name (a pending publisher never reserves a
name). It means the *configuration* is already in use — by our own other
package. The environment is the only field left to distinguish them, so it
carries the package name.

The constraint applies to *pending* publishers, which are 1:1 with a project
name. Once a project exists its publisher becomes a normal one, which several
projects may share — so this is purely a bootstrapping constraint. Distinct
environments sidestep it and give each package its own approval gate.

The GitHub environments are created automatically the first time the workflow
references them. Add a required-reviewer protection rule to either if uploads
should wait for manual approval.

## Error Handling

If any step fails:
- **Checks fail**: Stop and report issues, user must fix before continuing
- **Build fails**: Report error, check dependencies and pyproject.toml
- **Git conflicts**: Report and ask user to resolve manually
- **Version already exists**: Check if tag exists, confirm with user if override needed

## Post-Release Checklist

After user pushes to GitHub, remind them to:
- [ ] Verify GitHub Actions release workflow completes successfully
- [ ] Check GitHub Releases page for the new release (auto-created)
- [ ] Review auto-generated release notes and edit if needed
- [ ] Verify pre-release status is correct (if using -alpha/-beta/-rc suffixes)
- [ ] Verify both PyPI projects updated: <https://pypi.org/project/leech/> and
      <https://pypi.org/project/leech-core/>
- [ ] Smoke-test the published artifacts in a clean env:
      `uv run --isolated --with "leech[rust]==X.Y.Z" --no-project check-rust`
- [ ] Update any dependent projects or documentation
- [ ] Announce release (if applicable)
