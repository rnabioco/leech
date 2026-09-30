"""Registry-wide contract: every architecture honors ``num_out`` and rejects unknown kwargs."""

import pytest
import torch

from leech.models import MODEL_REGISTRY, get_model, requires_features, wide_features

SIGNAL_LEN = 200
KMER_LEN = 11
NUM_FEATURES = 5
BATCH = 3
MARGIN = 4

_SMALL = {"d_model": 32, "nhead": 4, "num_layers": 1}


def _kwargs(name: str, num_out: int) -> dict:
    kw: dict = {"signal_len": SIGNAL_LEN, "kmer_len": KMER_LEN, "num_out": num_out}
    if requires_features(name):
        kw["num_features"] = NUM_FEATURES
    if name.startswith("TransformerDwell"):
        kw.update(_SMALL)
    return kw


def _inputs(name: str):
    channels = 2 if "Residual" in name else 1
    signal = torch.randn(BATCH, channels, SIGNAL_LEN)
    if name.startswith("ConvLSTMRemora"):
        sequence = torch.randn(BATCH, 36, SIGNAL_LEN)  # signal_kmer is Remora's default
    else:
        sequence = torch.randn(BATCH, 4, KMER_LEN)
    if not requires_features(name):
        return (signal, sequence)
    width = KMER_LEN + (2 * MARGIN if wide_features(name) else 0)
    return (signal, sequence, torch.randn(BATCH, NUM_FEATURES, width))


@pytest.mark.parametrize("num_out", [1, 5])
@pytest.mark.parametrize("name", list(MODEL_REGISTRY.keys()))
def test_output_width_equals_num_out(name, num_out):
    model = get_model(name, **_kwargs(name, num_out)).eval()
    with torch.no_grad():
        out = model(*_inputs(name))
    assert out.shape == (BATCH, num_out)


# These still absorb extra keyword arguments on purpose: training passes every
# model the same kwargs (num_features, kmer_len, ...), and they ignore the ones
# they have no use for. Everything else must reject a name it does not know.
_LENIENT = {"ConvLSTMRemora", "ConvLSTMRemoraBase", "SignalCNN"}


@pytest.mark.parametrize("name", [n for n in MODEL_REGISTRY.keys() if n not in _LENIENT])
def test_unknown_kwarg_raises(name):
    with pytest.raises(TypeError):
        get_model(name, **_kwargs(name, 1), definitely_not_a_param=1)


@pytest.mark.parametrize("name", ["TransformerDwell", "TransformerDwellResidual"])
def test_transformer_towers_initialised_independently(name):
    model = get_model(name, **_kwargs(name, 1))
    sig = model.signal_transformer.layers[0].linear1.weight
    seq = model.seq_transformer.layers[0].linear1.weight
    assert not torch.equal(sig, seq)
