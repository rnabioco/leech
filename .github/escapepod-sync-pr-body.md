`rust/Cargo.toml` and `pyproject.toml` pin the same upstream, released in
lockstep from rnabioco/escapepod-rs, and leech drives **both** — `signal_refine.py`
through the Python binding, `refinement.rs` through the crate. A skew between
them lets the two prepare backends compute different dwells and different
level-derived features from the same read, which is issue #193: invisible for
four releases, because the arrays keep their shape and look plausible either way.

Opened by `.github/workflows/escapepod-sync.yml` because the pins disagreed.
Dependabot bumps the tag-pinned crate on its own schedule and has no way to know
a PyPI package in another manifest has to move with it.

`uv.lock`'s escapepod entry is rewritten from PyPI's release record rather than
by `uv lock`, which re-serializes the whole file (~670 changed lines, ~300
packages gaining emscripten markers) and would bury the one line worth reviewing.

## Verify before merging

This moves the pins and re-resolves the locks. It does not check that escapepod
still computes the same thing.

- **`tests/test_backend_parity.py` is the check that matters.** It compares
  every array in the npz across both prepare backends, which is exactly the
  divergence this guards against.
- **A compile is not enough.** `cargo check` passing says the API still exists,
  not that the values are unchanged.

Note: PRs opened with `GITHUB_TOKEN` do not start other workflows. Push an empty
commit, or close and reopen, to run CI.
