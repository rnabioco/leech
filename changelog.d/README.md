# Changelog fragments

`CHANGELOG.md` used to be edited by hand on every PR — every PR touched the
same `## [Unreleased]` section, so any two PRs open at once conflicted on it,
and the more PRs land in parallel the worse it gets (wave 1 of the 2026-09-13
review hit this on nearly every merge). This directory replaces that with one
fragment file per PR, assembled into `CHANGELOG.md` at release time by
[towncrier](https://towncrier.readthedocs.io/).

## Adding an entry

Create `changelog.d/<number>.<type>.md`, where `<number>` is the GitHub issue
or PR number the change belongs to, and `<type>` is one of:

| type | section |
|---|---|
| `added` | Added |
| `changed` | Changed |
| `deprecated` | Deprecated |
| `removed` | Removed |
| `fixed` | Fixed |
| `security` | Security |

Content is one or a few sentences, same voice as the existing entries in
`CHANGELOG.md`: lead with the user-visible change in bold, then the technical
detail. The issue/PR number is appended automatically — don't repeat it in
the text.

Example, `changelog.d/300.removed.md`:

```markdown
**`leech eval compare`, `leech eval importance` and `leech eval ablation`.**
All three imported a module that was never committed; nothing in CI ever
invoked them.
```

Multiple fragments may share a number (e.g. `300.removed.md` and
`300.fixed.md`) if one PR touches more than one section.

## At release time

`uv run towncrier build --version X.Y.Z` (part of `/release`) renders every
fragment in this directory into a new dated section at the top of
`CHANGELOG.md` and deletes the fragments it consumed. Nothing in this
directory should be edited by hand except to add a new fragment.
