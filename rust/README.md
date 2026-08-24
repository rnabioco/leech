# leech-core

Rust acceleration for [`leech`](https://github.com/rnabioco/leech) — *Learning
Enhanced Electrical Classifiers from Hanopore signals*.

This package ships the compiled `leech_core` extension module: POD5 I/O with a
process-global reader cache, signal-map refinement, per-base statistics,
`signal_kmer` encoding, and the batched chunk-extraction pipelines used by
`leech data prepare` and `leech predict`.

It is not useful on its own. Install it through `leech`:

```bash
pip install "leech[rust]"
```

`leech` runs without it — every accelerated path has a pure-Python fallback —
but preparation and inference are substantially faster with it installed.
`leech-core` is released in lockstep with `leech` and reports a matching
`__version__`; `check-rust` warns if the two ever diverge.

Wheels are published for manylinux x86_64 and aarch64. On other platforms pip
falls back to the sdist, which needs a Rust toolchain and network access to
github.com (the `escapepod-signal` dependency is fetched from git, not
crates.io).

MIT licensed. Source, issues, and documentation live in the
[`leech` repository](https://github.com/rnabioco/leech).
