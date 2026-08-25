"""Reading the CTC-CRF architecture config.

One reader, so the trainer, the exporter and every evaluation script derive
geometry the same way instead of each reaching into the dict for the keys it
happens to need. The dict-to-:class:`~leech.crf.encoder.EncoderConfig` step is
:func:`~leech.crf.encoder.encoder_config_from_toml`; this module is only about
*finding* and *parsing* the file.

The shipped config travels with the package (``leech/crf/configs/crf_ctc.toml``)
rather than living beside a corpus in scratch. Six scripts in escapepod-models
once read it from scratch alone, which meant a purge would have left trained
weights nobody could load — the architecture is part of the code, not part of
the data.

Torch-free at import time, like :mod:`leech.models.config_loader`.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

__all__ = ["CONFIG_DIR", "DEFAULT_CONFIG", "load_config"]

CONFIG_DIR = Path(__file__).parent / "configs"

#: The architecture the shipped barcode CRF models were trained with.
DEFAULT_CONFIG = CONFIG_DIR / "crf_ctc.toml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Parse a CRF architecture config; ``None`` means the packaged default.

    Returns the raw mapping rather than an ``EncoderConfig`` because callers
    need keys the encoder does not: ``labels.labels`` is the decode alphabet,
    and ``global_norm.state_len`` sizes the target as well as the encoder.
    """
    resolved = Path(path) if path is not None else DEFAULT_CONFIG
    if not resolved.is_file():
        raise FileNotFoundError(f"CRF config not found: {resolved}")
    with resolved.open("rb") as fh:
        return tomllib.load(fh)
