"""
Micro-benchmark: per-call argument-conversion overhead of the Rust extraction
entry points, with the k-mer refinement table attached (issue #259).

Before this change, every ``extract_training_chunks`` / ``extract_inference_chunks``
call converted the caller's ``dict[str, float]`` k-mer table to a Rust
``HashMap<String, f64>`` itself, under the GIL and before ``py.detach`` --
measured (with zero reads, so only the conversion runs) at 51.8ms min / 71.7ms
median for the real 262,144-entry 9-mer table. That ran on *every batch* of
every prepare/predict run, serializing the ``ThreadPoolExecutor`` workers
``_iter_rust_batches`` exists to overlap.

``KmerLevels`` (built once via ``leech._rust_accel.make_kmer_levels``) moves
that conversion out of the call: this test builds the handle once, outside
the timed loop, and asserts what's left -- borrowing the handle and marshaling
the (empty, in this benchmark) read lists -- is well under the old per-call
cost.

Marked ``slow`` (run with ``pytest --slow``) because it loads the full
262,144-entry production-scale k-mer table fixture; it does not need real
POD5 reads (zero reads isolates exactly the argument-conversion cost, which is
how the issue's own before/after numbers were measured).
"""

import statistics
import time

import pytest
from conftest import LEVELS_FILE, TRNA_FIXTURES_AVAILABLE, TRNA_POD5

pytest.importorskip("leech_core")

from leech._rust_accel import (  # noqa: E402
    HAS_RUST,
    _rs_extract_training_chunks,
    make_kmer_levels,
)
from leech.signal_refine import load_kmer_table  # noqa: E402

pytestmark = pytest.mark.slow

# Generous on purpose: this is a regression guard against the ~50-70ms/call
# the old per-call `dict -> HashMap` conversion cost, not a tight perf target.
MAX_PER_CALL_MS = 5.0
N_CALLS = 200


@pytest.mark.skipif(not TRNA_FIXTURES_AVAILABLE, reason="tRNA test fixtures not available")
def test_kmer_table_argument_conversion_is_cheap_once_built():
    """Per-call overhead with a pre-built ``KmerLevels`` handle stays well
    under the old per-call `dict` conversion cost.

    Zero reads (empty lists everywhere) isolates the argument-conversion cost
    from POD5 I/O and per-read processing, matching how the issue's own
    51.8ms/71.7ms numbers were measured.
    """
    if not HAS_RUST or _rs_extract_training_chunks is None:
        pytest.skip("leech_core Rust acceleration not available")

    kmer_to_level, kmer_len = load_kmer_table(LEVELS_FILE)
    assert len(kmer_to_level) == 262_144, "fixture table should be the full 9-mer table"

    # Built ONCE, outside the timed loop -- this is the fix.
    kmer_levels = make_kmer_levels(kmer_to_level)
    assert kmer_levels is not None

    def _call():
        return _rs_extract_training_chunks(
            pod5_path=str(TRNA_POD5),
            read_ids=[],
            sequences=[],
            mv_strides=[],
            mv_arrays=[],
            num_samples_list=[],
            trim_offsets=[],
            signal_context_left=200,
            signal_context_right=200,
            kmer_context=5,
            motif_positions=[],
            signal_len=400,
            compute_features=True,
            refine_signal_map=True,
            kmer_table=kmer_levels,
            kmer_len=kmer_len,
            kmer_center_idx=kmer_len // 2,
            refine_half_bandwidth=5,
            refine_scale_iters=2,
        )

    # Warm up (first call pays one-time page-fault / cache-warming cost that
    # has nothing to do with argument conversion).
    # `(chunks, n_missing_from_pod5)` since #325 -- no read ids submitted, so
    # nothing to extract and nothing to be missing.
    assert _call() == ([], 0)

    timings_ms = []
    for _ in range(N_CALLS):
        t0 = time.perf_counter()
        _call()
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    min_ms = min(timings_ms)
    median_ms = statistics.median(timings_ms)
    print(
        f"\nKmerLevels-attached call: min={min_ms:.3f}ms median={median_ms:.3f}ms "
        f"over {N_CALLS} calls (old per-call dict conversion: "
        f"51.8ms min / 71.7ms median)"
    )

    assert median_ms < MAX_PER_CALL_MS, (
        f"median per-call time {median_ms:.3f}ms exceeds {MAX_PER_CALL_MS}ms -- "
        "the k-mer table may be getting re-converted per call again"
    )
