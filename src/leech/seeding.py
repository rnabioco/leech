"""Random-seed setup, torch-free at import time.

``leech.model_loading`` re-exports :func:`setup_random_seed` from here so
existing callers (training code) are unaffected -- import it from either
module and get the same function.

Why this module exists (issue #265): ``commands/prepare.py``,
``preparation/orchestrator.py`` and ``splitting/splitter.py`` all call
``setup_random_seed`` to make a CPU-only data-preparation run reproducible,
with no model or GPU involved anywhere in that path. Before this module
existed, the only place to import the function from was
``leech.model_loading``, whose module scope does ``import torch`` -- so
merely importing the name pulled in torch, seconds of startup cost on a data
command and, worse, ahead of the ``multiprocessing.Pool`` fork
``preparation/parallel.py`` does for the worker-process backend. Forking
after torch has initialized its OpenMP thread pools is a known hang hazard.

This module imports only ``random``, ``numpy`` and stdlib -- never torch at
module scope. ``setup_random_seed`` still seeds torch when it is actually
called, but does the import lazily, inside the function body, right before
it is needed.
"""

import logging
import random
from pathlib import Path

import numpy as np

from leech.constants import generate_random_seed

logger = logging.getLogger("leech.seeding")


def setup_random_seed(seed: int | None, output_dir: Path | None = None) -> int:
    """Setup random seed for reproducibility and optionally save to file.

    Args:
        seed: Random seed value, or None to generate one
        output_dir: Directory to save seed.txt file, or None to skip saving

    Returns:
        The seed value used
    """
    # Generate if needed
    if seed is None:
        seed = generate_random_seed()
        logger.info(f"Generated random seed: {seed}")
    else:
        logger.info(f"Using provided seed: {seed}")

    # Set for all libraries
    random.seed(seed)
    np.random.seed(seed)

    # Lazy: keeps this module torch-free at import time. See module docstring
    # -- a caller that only needs numpy/random reproducibility (data prepare)
    # never pays for a torch import or risks forking a worker pool after one.
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Save if requested
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        seed_file = output_dir / "seed.txt"
        with open(seed_file, "w") as f:
            f.write(f"{seed}\n")
        logger.info(f"Saved seed to {seed_file}")

    return seed
