# ADR 0006: Memory-Mapped Chunk Corpora

**Status:** Accepted (mapping rejected for the training read path; sequential
shard streaming adopted as the direction)

**Date:** 2026-08-24

## Context

[#211](https://github.com/rnabioco/leech/issues/211) listed four fixes for
`LeechDataset` peaking at 116 GB on a 41 GB corpus. Three are done (PR #212):
the `torch.stack` transient is gone, the arrays are read in row blocks instead
of materialised, and the metadata is columnar. What remains is the output
tensors themselves — for that corpus, signals `(6668328, 2, 540)` fp32 =
28.8 GB and features `(6668328, 12, 21)` fp32 = 6.7 GB, since
`TCNDwellResidualLN` is a wide-feature model and keeps the full 21-base window.
Roughly 36 GB, resident for the whole run.

Item 4 of the issue proposed mapping the file instead: under `--no-compress`
every member is written *stored*, so each could be `np.memmap`'d at its zip
offset and `__getitem__` could slice from disk. The pickled `seq_to_sig_maps`
member blocked that; PR #212 replaced it with a CSR pair, so **every member of a
freshly written uncompressed corpus is now fixed-shape, non-object, and
mappable in place**. The blocker is gone. The question left is whether mapping
is the right thing to do, which the issue explicitly flagged as a design call
rather than a patch.

### What the training loop actually asks for

Per chunk the loader touches three members at unrelated offsets — signal,
residual, features — about 5.3 KB in total. The reported production loader
wants ~10,900 chunks/s.

| Access pattern | What that costs |
|---|---|
| Sequential | 10,900 × 5.3 KB = **58 MB/s** |
| Shuffled (mapped) | ≥3 page faults per chunk, more when a 2160-byte row straddles a page: **40–65k IOPS** |

Those are wildly different asks of the same file, and shuffling is the only
difference.

### Measured on this cluster

An `rna` compute node, against a 1.83 GB corpus on BeeGFS. Client page cache
dropped with `posix_fadvise(POSIX_FADV_DONTNEED)` before each cold run:

| | measured |
|---|---|
| sequential read, cold | **724 MB/s** (single stream) |
| random 4K `pread`, 1 thread | 4,712 IOPS |
| random 4K `pread`, 16 threads | 152,000 IOPS aggregate |
| **mmap page fault, 1 thread, cold** | **1,668 rows/s** (0.60 ms each) |
| mmap page fault, warm | 342,000 rows/s |

**Treat the random numbers as upper bounds.** `POSIX_FADV_DONTNEED` drops the
*client's* cache; it cannot drop the BeeGFS servers', and the node was idle.
The production figure in #211 — ~113 random IOPS per thread against a loaded
filesystem — is three orders of magnitude below the idle-node number here, and
is the one to plan against. That spread *is* the finding: random-read
performance on this storage is a property of who else is using it.

The gap between the two mmap rows is the whole story. A mapped corpus runs at
RAM speed while the page cache holds it, and at 1,668 chunks/s per thread when
it does not — and not holding it in RAM is the entire reason for mapping.

Node-local staging does not rescue it here: the compute nodes' local disk is
rotational (`lsblk ROTA=1`), and `/dev/shm` is a 377 GB tmpfs, which is RAM
with extra steps — staging there to "save memory" spends exactly the memory it
claims to save.

## Decision

**Do not map the corpus for the shuffled training read path.** Keep the tensors
resident, and pursue the two directions below instead.

### 1. Sequential shard streaming with a shuffle buffer (the real answer)

The arithmetic above says the corpus streams at **12× the rate training
consumes it** (724 MB/s against 58 MB/s), and a full epoch of pure sequential
I/O over 41 GB is 57 seconds. Only the shuffle makes disk-resident training
hard, and a shuffle buffer is the standard trade: read shards sequentially,
shuffle within a window of B chunks, yield from the window.

- RAM for the corpus drops from ~36 GB to `B × 5.3 KB` — 1.1 GB at B = 200,000
  chunks (3% of the corpus), plus prefetch.
- `iter_npz_row_blocks` from PR #212 is already the primitive: sequential row
  blocks with a byte budget, one block resident.
- Randomness is approximate rather than exact. Interleaving several shards into
  the window recovers most of it.

**Constraints that must be designed around, not discovered later:**

- `--balance-groups` and `--oversample-minority` build a `WeightedRandomSampler`
  over the whole index, which assumes global random access. The counts they
  need are now free (`ChunkTable.values("source_group")` is a column), but the
  sampling itself has to move inside the buffer — per-window weighted draws or
  rejection sampling against the global rate.
- `--feature-noise-scale` derives per-channel stds from the whole feature
  tensor. Streaming needs a two-pass or running estimate.
- Workers must own disjoint shards, and the shuffle must be seedable, or runs
  stop being reproducible.
- Validation and eval read in fixed order and need none of this.

### 2. Halve the dominant term first (cheaper, orthogonal, no loader change)

Signals are 28.8 GB of the 36 GB. They arrive as 16-bit ADC counts and are
stored normalised; **fp16 holds normalised signal to about 1e-3 at the
magnitudes involved**, which is far below the noise the model is being asked to
see through. Storing the signal tensor as fp16 and casting per batch takes the
resident corpus from ~36 GB to ~21 GB for a few lines and no change to how the
data is read. This should be tried before anything more ambitious, and needs an
accuracy check against a trained model, not just a memory measurement.

### Where mapping *is* the right tool

For the sequential passes — feature-std computation, label and group tallies,
anything that touches every chunk once in order — mapping is fine, and so is
the block reader, which is already there and does not depend on cache state.
Neither is worth adding for those.

## Consequences

**Positive**

- No one spends a sprint building a mapped loader whose throughput is a
  function of cluster load, and which degrades to 1,668 chunks/s exactly when
  RAM is scarce enough to have wanted it.
- The direction that does work reuses the row-block reader already in the tree.
- fp16 gives most of the remaining win for a fraction of the effort.

**Negative**

- The 36 GB of tensors stays for now. Corpora much beyond the current one still
  need a bigger memory request until the streaming loader exists.
- Shuffle-buffer training is a real project, with the sampler constraints
  above, not a patch.

**Neutral**

- The mapping option stays open: PR #212's CSR change means an uncompressed
  corpus is mappable today, so if the storage picture changes — node-local
  NVMe, or a corpus small enough to stay in page cache — the experiment is a
  short one. Nothing in this decision has to be undone to run it.

## Alternatives Considered

**Map the members and slice in `__getitem__` (issue item 4 as written).**
Rejected above: 1,668 chunks/s per thread cold on an idle node, ~113 IOPS per
thread under production load, against a loader wanting 10,900 chunks/s.

**Stage to node-local disk, then map.** The usual fix for this problem, and it
is the reason the option is not dead in general — but these nodes have
rotational local disks. Revisit if the hardware changes.

**`/dev/shm`.** tmpfs is RAM. Mapping from it does not reduce memory, it
relocates it out of the process's accounting where the scheduler can no longer
see it.

**Keep compressed npz and map anyway.** Not possible: deflate members have no
byte-addressable layout. `--no-compress` is a precondition for any mapping
work, and costs disk (the reported corpus is 41 GB stored).

**Convert to a format built for this (webdataset, FFCV, Arrow/Parquet).**
Would bring sharded sequential reads and a shuffle buffer with it rather than
hand-rolling them. Rejected for now only because it is a format migration on
top of a corpus that is already produced, validated and parity-tested; the
shuffle-buffer design above can be built against the existing npz and revisited
if it grows past what one module should own.

## Notes

Measurements: `srun -p rna -c 16`, 1.83 GB synthetic corpus with the shape of
the one in #211 (540-sample signals, residual channel, 12×21 features).
Reproduce with `os.posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` before each
cold pass, and read the caveat above about server-side caching before quoting
any of the random-access numbers.
