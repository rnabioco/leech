"""
Tests for model architectures.

Covers the hand-written zoo (TransformerDwell, ConvOnly, ResNetDwell, ...)
plus the config-driven layer registry and the TOML-declared architectures that
replaced the ConvLSTM and TCN families.
"""

import pytest
import torch

from leech.models import (
    MODEL_REGISTRY,
    ConvLSTMBase,
    ConvLSTMDwell,
    ConvOnly,
    ResNetDwell,
    TCNDwell,
    TransformerDwell,
    get_model,
)


def small_model_kwargs(model_name: str, signal_len: int = 100, kmer_len: int = 11) -> dict:
    """Constructor kwargs that build a small instance of any registered model.

    Selection is by architecture *family* (prefix), so newly registered names
    are covered automatically instead of silently reusing another model's
    kwargs.
    """
    kwargs: dict = {"signal_len": signal_len, "kmer_len": kmer_len}
    if model_name.startswith("ConvLSTMRemora"):
        kwargs["size"] = 32
    elif model_name.startswith("ConvLSTM"):
        kwargs.update({"conv_channels": [4, 16, 64], "lstm_hidden": 32})
    elif model_name.startswith("Transformer"):
        kwargs.update({"d_model": 64, "nhead": 4, "num_layers": 2})
    elif model_name.startswith("TCN"):
        kwargs.update({"hidden_channels": 32, "num_layers": 3, "kernel_size": 3})
    elif model_name in ("ConvOnly", "ResNetDwell"):
        kwargs["base_channels"] = 8
    if "Dwell" in model_name or model_name in ("ConvOnly", "ConvLSTMRemora"):
        kwargs["num_features"] = 5
    return kwargs


class TestModelRegistry:
    """Test model registry and get_model function."""

    def test_model_registry_complete(self):
        """Test that MODEL_REGISTRY contains all expected models."""
        # Smoke-check: registry is non-empty and contains core models
        assert len(MODEL_REGISTRY) >= 10
        for name in ("ConvLSTMDwell", "ConvLSTMBase", "ConvLSTMRemora", "TCNDwell", "ResNetDwell"):
            assert name in MODEL_REGISTRY, f"Core model {name} missing from registry"

    def test_registry_listing_is_torch_free(self):
        """Listing model names must not import torch (keeps CLI help fast).

        The CLI renders `--model` choices from MODEL_REGISTRY.keys(); if that
        ever pulls in torch again, `leech model train -h` regresses to ~15s.
        """
        import subprocess
        import sys

        code = (
            "import sys; from leech.models import MODEL_REGISTRY; "
            "sorted(MODEL_REGISTRY.keys()); "
            "assert 'torch' not in sys.modules, 'listing model names imported torch'"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_get_model_valid(self, model_config):
        """Test getting a model by name."""
        model = get_model("ConvLSTMDwell", **model_config)
        assert isinstance(model, ConvLSTMDwell)

    def test_get_model_invalid(self, model_config):
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model("InvalidModel", **model_config)


class TestConvLSTMBase:
    """Test ConvLSTMBase model."""

    def test_initialization(self, model_config):
        """Test model initialization."""
        model = ConvLSTMBase(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            conv_channels=model_config["conv_channels"],
            lstm_hidden=model_config["lstm_hidden"],
            dropout=model_config["dropout"],
        )
        assert model.signal_len == model_config["signal_len"]
        assert model.kmer_len == model_config["kmer_len"]
        assert model.lstm_hidden == model_config["lstm_hidden"]

    def test_forward_pass(self, model_config):
        """Test forward pass with valid inputs."""
        # ConvLSTMBase doesn't take num_features
        model = ConvLSTMBase(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            conv_channels=model_config["conv_channels"],
            lstm_hidden=model_config["lstm_hidden"],
            dropout=model_config["dropout"],
        )
        model.eval()

        batch_size = 4
        signal = torch.randn(batch_size, model_config["signal_len"])
        sequence = torch.randn(batch_size, 4, model_config["kmer_len"])

        with torch.no_grad():
            output = model(signal, sequence)

        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_predict_proba(self, model_config):
        """Test probability prediction."""
        model = ConvLSTMBase(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            conv_channels=model_config["conv_channels"],
            lstm_hidden=model_config["lstm_hidden"],
            dropout=model_config["dropout"],
        )
        model.eval()

        batch_size = 4
        signal = torch.randn(batch_size, model_config["signal_len"])
        sequence = torch.randn(batch_size, 4, model_config["kmer_len"])

        with torch.no_grad():
            probs = model.predict_proba(signal, sequence)

        assert probs.shape == (batch_size, 1)
        assert torch.all((probs >= 0) & (probs <= 1))

    def test_different_batch_sizes(self, model_config):
        """Test model with different batch sizes."""
        model = ConvLSTMBase(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            conv_channels=model_config["conv_channels"],
            lstm_hidden=model_config["lstm_hidden"],
            dropout=model_config["dropout"],
        )
        model.eval()

        for batch_size in [1, 2, 8, 16]:
            signal = torch.randn(batch_size, model_config["signal_len"])
            sequence = torch.randn(batch_size, 4, model_config["kmer_len"])

            with torch.no_grad():
                output = model(signal, sequence)

            assert output.shape == (batch_size, 1)


class TestConvLSTMDwell:
    """Test ConvLSTMDwell model."""

    def test_initialization(self, model_config):
        """Test model initialization."""
        model = ConvLSTMDwell(**model_config)
        assert model.signal_len == model_config["signal_len"]
        assert model.kmer_len == model_config["kmer_len"]
        assert model.num_features == model_config["num_features"]

    def test_forward_pass(self, model_config):
        """Test forward pass with valid inputs."""
        model = ConvLSTMDwell(**model_config)
        model.eval()

        batch_size = 4
        signal = torch.randn(batch_size, model_config["signal_len"])
        sequence = torch.randn(batch_size, 4, model_config["kmer_len"])
        features = torch.randn(batch_size, model_config["num_features"], model_config["kmer_len"])

        with torch.no_grad():
            output = model(signal, sequence, features)

        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()

    def test_predict_proba(self, model_config):
        """Test probability prediction."""
        model = ConvLSTMDwell(**model_config)
        model.eval()

        batch_size = 4
        signal = torch.randn(batch_size, model_config["signal_len"])
        sequence = torch.randn(batch_size, 4, model_config["kmer_len"])
        features = torch.randn(batch_size, model_config["num_features"], model_config["kmer_len"])

        with torch.no_grad():
            probs = model.predict_proba(signal, sequence, features)

        assert probs.shape == (batch_size, 1)
        assert torch.all((probs >= 0) & (probs <= 1))

    def test_three_branch_architecture(self, model_config):
        """Test that all three branches (signal, sequence, features) are used."""
        model = ConvLSTMDwell(**model_config)
        assert hasattr(model, "signal_branch")
        assert hasattr(model, "sequence_branch")
        assert hasattr(model, "feature_branch")


class TestTransformerDwell:
    """Test TransformerDwell model."""

    def test_initialization(self, model_config):
        """Test model initialization."""
        model = TransformerDwell(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            num_features=model_config["num_features"],
            d_model=64,  # Smaller for testing
            nhead=4,
            num_layers=2,
            dropout=model_config["dropout"],
        )
        assert model.signal_len == model_config["signal_len"]
        assert model.d_model == 64

    def test_forward_pass(self, model_config):
        """Test forward pass with wide features (cross-attention over dwell margin)."""
        dwell_margin = 5
        model = TransformerDwell(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            num_features=model_config["num_features"],
            dwell_margin=dwell_margin,
            d_model=64,
            nhead=4,
            num_layers=2,
        )
        model.eval()

        batch_size = 4
        feat_len = model_config["kmer_len"] + 2 * dwell_margin
        signal = torch.randn(batch_size, model_config["signal_len"])
        sequence = torch.randn(batch_size, 4, model_config["kmer_len"])
        features = torch.randn(batch_size, model_config["num_features"], feat_len)

        with torch.no_grad():
            output = model(signal, sequence, features)

        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()

    def test_positional_encoding(self, model_config):
        """Test that positional encoding is applied."""
        model = TransformerDwell(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            num_features=model_config["num_features"],
            d_model=64,
            nhead=4,
            num_layers=2,
        )
        assert hasattr(model, "pos_encoding")


class TestConvOnly:
    """Test ConvOnly model."""

    def test_initialization(self, model_config):
        """Test model initialization."""
        model = ConvOnly(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            num_features=model_config["num_features"],
            base_channels=8,  # Smaller for testing
            num_blocks=2,
            dropout=model_config["dropout"],
        )
        assert model.signal_len == model_config["signal_len"]

    def test_forward_pass(self, model_config):
        """Test forward pass with wide features (cross-attention over dwell margin)."""
        dwell_margin = 5
        model = ConvOnly(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            num_features=model_config["num_features"],
            dwell_margin=dwell_margin,
            base_channels=8,
            num_blocks=2,
        )
        model.eval()

        batch_size = 4
        feat_len = model_config["kmer_len"] + 2 * dwell_margin
        signal = torch.randn(batch_size, model_config["signal_len"])
        sequence = torch.randn(batch_size, 4, model_config["kmer_len"])
        features = torch.randn(batch_size, model_config["num_features"], feat_len)

        with torch.no_grad():
            output = model(signal, sequence, features)

        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()

    def test_inception_blocks(self, model_config):
        """Test that InceptionBlocks are created."""
        model = ConvOnly(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            num_features=model_config["num_features"],
            base_channels=8,
            num_blocks=2,
        )
        assert len(model.signal_conv) == 2  # num_blocks
        assert len(model.seq_conv) == 2
        assert hasattr(model, "feature_branch")


class TestTCNDwell:
    """Test TCNDwell model."""

    def test_initialization(self, model_config):
        """Test model initialization."""
        model = TCNDwell(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            num_features=model_config["num_features"],
            hidden_channels=16,  # Smaller for testing
            num_layers=4,  # Smaller for testing
            kernel_size=3,
            dropout=model_config["dropout"],
        )
        assert model.signal_len == model_config["signal_len"]

    def test_forward_pass(self, model_config):
        """Test forward pass with wide features (cross-attention over dwell margin)."""
        dwell_margin = 5
        model = TCNDwell(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            num_features=model_config["num_features"],
            dwell_margin=dwell_margin,
            hidden_channels=16,
            num_layers=4,
            kernel_size=3,
        )
        model.eval()

        batch_size = 4
        feat_len = model_config["kmer_len"] + 2 * dwell_margin
        signal = torch.randn(batch_size, model_config["signal_len"])
        sequence = torch.randn(batch_size, 4, model_config["kmer_len"])
        features = torch.randn(batch_size, model_config["num_features"], feat_len)

        with torch.no_grad():
            output = model(signal, sequence, features)

        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()


class TestResNetDwell:
    """Test ResNetDwell model."""

    def test_initialization(self, model_config):
        """Test model initialization."""
        model = ResNetDwell(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            num_features=model_config["num_features"],
            base_channels=8,  # Smaller for testing
            dropout=model_config["dropout"],
        )
        assert model.signal_len == model_config["signal_len"]

    def test_forward_pass(self, model_config):
        """Test forward pass with wide features (cross-attention over dwell margin)."""
        dwell_margin = 5
        model = ResNetDwell(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            num_features=model_config["num_features"],
            dwell_margin=dwell_margin,
            base_channels=8,
        )
        model.eval()

        batch_size = 4
        feat_len = model_config["kmer_len"] + 2 * dwell_margin
        signal = torch.randn(batch_size, model_config["signal_len"])
        sequence = torch.randn(batch_size, 4, model_config["kmer_len"])
        features = torch.randn(batch_size, model_config["num_features"], feat_len)

        with torch.no_grad():
            output = model(signal, sequence, features)

        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()

    def test_residual_blocks(self, model_config):
        """Test that ResidualBlocks are created."""
        model = ResNetDwell(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            num_features=model_config["num_features"],
            base_channels=8,
        )
        # Check that branches exist
        assert hasattr(model, "signal_resnet")
        assert hasattr(model, "seq_resnet")
        assert hasattr(model, "feature_branch")


class TestModelComparisons:
    """Test comparisons across all models."""

    @pytest.mark.parametrize(
        "model_name,requires_features",
        [
            ("ConvLSTMBase", False),
            ("ConvLSTMDwell", True),
            ("ConvLSTMDwellGNAttn", True),
            ("ConvLSTMDwellLNAttn", True),
            ("TransformerDwell", True),
            ("ConvOnly", True),
            ("TCNDwell", True),
            ("TCNDwellGN", True),
            ("TCNDwellLN", True),
            ("ResNetDwell", True),
        ],
    )
    def test_all_models_forward(self, model_name, requires_features):
        """Test that all models can do forward pass."""
        from leech.models.inference_wrapper import ModelInferenceWrapper

        # Base config for all models
        dwell_margin = 5
        config = {
            "signal_len": 100,
            "kmer_len": 11,
        }

        # Model-specific configs
        if model_name == "ConvLSTMBase":
            config.update({"conv_channels": [4, 16, 32], "lstm_hidden": 16})
        elif model_name == "ConvLSTMDwell":
            config.update({"num_features": 5, "conv_channels": [4, 16, 32], "lstm_hidden": 16})
        elif model_name in ("ConvLSTMDwellGNAttn", "ConvLSTMDwellLNAttn"):
            config.update(
                {
                    "num_features": 5,
                    "dwell_margin": dwell_margin,
                    "conv_channels": [4, 16, 32],
                    "lstm_hidden": 16,
                }
            )
        elif model_name == "TransformerDwell":
            config.update(
                {
                    "num_features": 5,
                    "dwell_margin": dwell_margin,
                    "d_model": 32,
                    "nhead": 4,
                    "num_layers": 1,
                }
            )
        elif model_name == "ConvOnly":
            config.update({"num_features": 5, "dwell_margin": dwell_margin, "base_channels": 4})
        elif model_name == "ResNetDwell":
            config.update({"num_features": 5, "dwell_margin": dwell_margin, "base_channels": 4})
        elif model_name in ("TCNDwell", "TCNDwellGN", "TCNDwellLN"):
            config.update(
                {
                    "num_features": 5,
                    "dwell_margin": dwell_margin,
                    "hidden_channels": 16,
                    "num_layers": 2,
                    "kernel_size": 3,
                }
            )

        model = get_model(model_name, **config)
        model.eval()

        batch_size = 2
        signal = torch.randn(batch_size, config["signal_len"])
        sequence = torch.randn(batch_size, 4, config["kmer_len"])
        if requires_features:
            wide = model_name in ModelInferenceWrapper.WIDE_FEATURE_MODELS
            feat_len = config["kmer_len"] + 2 * dwell_margin if wide else config["kmer_len"]
            features = torch.randn(batch_size, config["num_features"], feat_len)
        else:
            features = None

        with torch.no_grad():
            if requires_features:
                output = model(signal, sequence, features)
            else:
                output = model(signal, sequence)

        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.slow
    def test_all_models_exportable(self):
        """Test that a complex model can be exported with torch.export."""
        from leech.model_export import export_model

        config = {
            "signal_len": 100,
            "kmer_len": 11,
            "num_features": 5,
            "conv_channels": [4, 16, 32],
            "lstm_hidden": 16,
        }

        model = get_model("ConvLSTMDwell", **config)
        full_config = {"model_name": "ConvLSTMDwell", **config}
        ep = export_model(model, full_config)
        exported_module = ep.module()

        batch_size = 2
        signal = torch.randn(batch_size, config["signal_len"])
        sequence = torch.randn(batch_size, 4, config["kmer_len"])
        features = torch.randn(batch_size, config["num_features"], config["kmer_len"])

        with torch.no_grad():
            output = exported_module(signal, sequence, features)

        assert output.shape == (batch_size, 1)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.parametrize("model_name", sorted(MODEL_REGISTRY.keys()))
    def test_model_parameter_counts(self, model_name):
        """Test that models have reasonable parameter counts."""
        model = get_model(model_name, **small_model_kwargs(model_name, signal_len=400))
        param_count = sum(p.numel() for p in model.parameters())

        # Models should have between 1k and 10M parameters
        assert 1_000 < param_count < 10_000_000, f"{model_name} has {param_count} parameters"


class TestLayerRegistry:
    """Bonito-style layer registry: to_dict/from_dict round-trips."""

    def test_register_and_lookup(self):
        from leech.models import nn as leech_nn

        for expected in ("serial", "stack", "parallel", "graph", "concat", "bilstm", "mlphead"):
            assert expected in leech_nn.layers
            assert leech_nn.layers[expected].name == expected

    def test_serial_round_trip(self):
        from leech.models import nn as leech_nn

        model = leech_nn.Serial(
            [
                leech_nn.Linear(8, 16),
                leech_nn.Linear(16, 4),
            ]
        )
        spec = leech_nn.to_dict(model)
        assert spec["type"] == "serial"
        rebuilt = leech_nn.from_dict(spec)
        assert list(rebuilt.state_dict()) == list(model.state_dict())
        rebuilt.load_state_dict(model.state_dict())
        x = torch.randn(3, 8)
        assert torch.equal(model(x), rebuilt(x))

    def test_stack_round_trip(self):
        from leech.models import nn as leech_nn

        model = leech_nn.Stack([leech_nn.Linear(8, 8) for _ in range(3)])
        spec = leech_nn.to_dict(model)
        assert spec == {
            "type": "stack",
            "layer": {"type": "linear", "in_features": 8, "out_features": 8, "bias": True},
            "depth": 3,
        }
        rebuilt = leech_nn.from_dict(spec)
        assert len(rebuilt) == 3
        assert list(rebuilt.state_dict()) == list(model.state_dict())

    def test_parallel_fans_out_and_concatenates(self):
        """Parallel is the nestable multi-branch primitive."""
        from leech.models import nn as leech_nn

        model = leech_nn.Parallel(
            {
                "a": leech_nn.SignalBranch(in_channels=1, conv_channels=[2, 4, 8]),
                "b": leech_nn.SignalBranch(in_channels=1, conv_channels=[2, 4, 6]),
            }
        )
        x = torch.randn(2, 40)
        out = model(x)
        # branches must stay temporally aligned: same length, channels concatenated
        assert out.shape == (2, 8 + 6, 40)

        spec = leech_nn.to_dict(model)
        assert spec["type"] == "parallel"
        assert set(spec["sublayers"]) == {"a", "b"}
        rebuilt = leech_nn.from_dict(spec)
        assert list(rebuilt.state_dict()) == list(model.state_dict())
        rebuilt.load_state_dict(model.state_dict())
        assert torch.equal(model(x), rebuilt(x))

    def test_graph_round_trip(self):
        """A config-built Graph survives to_dict -> from_dict."""
        from leech.models import nn as leech_nn

        model = get_model(
            "ConvLSTMDwell",
            signal_len=100,
            kmer_len=11,
            num_features=5,
            conv_channels=[4, 16, 32],
            lstm_hidden=16,
        )
        spec = leech_nn.to_dict(model)
        assert spec["type"] == "graph"
        assert [node["name"] for node in spec["nodes"]][:3] == [
            "signal_branch",
            "sequence_branch",
            "feature_branch",
        ]

        rebuilt = leech_nn.from_dict(spec)
        assert list(rebuilt.state_dict()) == list(model.state_dict())
        rebuilt.load_state_dict(model.state_dict())
        model.eval()
        rebuilt.eval()

        signal = torch.randn(2, 100)
        sequence = torch.randn(2, 4, 11)
        features = torch.randn(2, 5, 11)
        with torch.no_grad():
            assert torch.equal(
                model(signal, sequence, features), rebuilt(signal, sequence, features)
            )

    def test_graph_build_order_round_trip(self):
        """build_order decouples module-construction order from dataflow order."""
        from leech.models import nn as leech_nn

        model = get_model(
            "TCNDwellResidualDwellAttn",
            signal_len=100,
            kmer_len=11,
            num_features=5,
            hidden_channels=16,
            num_layers=2,
        )
        spec = leech_nn.to_dict(model)
        assert spec["type"] == "graph"
        # dwell_* nodes execute before pool_linear but are installed after it
        exec_order = [node["name"] for node in spec["nodes"]]
        assert exec_order.index("dwell_attn_norm") < exec_order.index("pool_linear")
        assert spec["build_order"].index("dwell_attn_norm") > spec["build_order"].index(
            "pool_linear"
        )

        rebuilt = leech_nn.from_dict(spec)
        assert list(rebuilt.state_dict()) == list(model.state_dict())
        rebuilt.load_state_dict(model.state_dict())
        model.eval()
        rebuilt.eval()
        signal = torch.randn(2, 2, 100)
        sequence = torch.randn(2, 4, 11)
        features = torch.randn(2, 5, 21)
        with torch.no_grad():
            assert torch.equal(
                model(signal, sequence, features), rebuilt(signal, sequence, features)
            )

    def test_graph_build_order_must_be_a_permutation(self):
        from leech.models import nn as leech_nn

        nodes = [
            {"name": "a", "layer": leech_nn.Linear(4, 4), "inputs": ["signal"], "out": "x"},
            {"name": "b", "layer": leech_nn.Linear(4, 4), "inputs": ["x"], "out": "logits"},
        ]
        with pytest.raises(ValueError, match="permutation"):
            leech_nn.Graph(nodes, build_order=["a"])

    def test_unknown_layer_type_raises(self):
        from leech.models import nn as leech_nn

        with pytest.raises(KeyError, match="Unknown layer type"):
            leech_nn.from_dict({"type": "definitely_not_a_layer"})


# Config-built name -> (reference class name, has dwell features, wide features)
_CONVLSTM_PARITY = [
    ("ConvLSTMBase", False, False),
    ("ConvLSTMBaseBN", False, False),
    ("ConvLSTMDwell", True, False),
    ("ConvLSTMDwellBN", True, False),
    ("ConvLSTMBaseAttn", False, False),
    ("ConvLSTMBaseBNAttn", False, False),
    ("ConvLSTMDwellAttn", True, True),
    ("ConvLSTMDwellBNAttn", True, True),
    ("ConvLSTMDwellGNAttn", True, True),
    ("ConvLSTMDwellLNAttn", True, True),
]


class TestConfigDrivenParity:
    """Config-built ConvLSTM models must match the classes they replaced."""

    DWELL_MARGIN = 5
    BASE = {
        "signal_len": 100,
        "kmer_len": 11,
        "conv_channels": [4, 16, 32],
        "lstm_hidden": 16,
    }

    def _kwargs(self, has_features, wide):
        kwargs = dict(self.BASE)
        if has_features:
            kwargs["num_features"] = 5
        if wide:
            kwargs["dwell_margin"] = self.DWELL_MARGIN
        return kwargs

    @pytest.mark.parametrize("name,has_features,wide", _CONVLSTM_PARITY)
    def test_state_dict_keys_unchanged(self, name, has_features, wide):
        """Checkpoint compatibility: parameter names must not drift."""
        import reference_conv_lstm

        kwargs = self._kwargs(has_features, wide)
        reference = getattr(reference_conv_lstm, name)(**kwargs)
        built = get_model(name, **kwargs)
        assert list(built.state_dict()) == list(reference.state_dict())

    @pytest.mark.parametrize("name,has_features,wide", _CONVLSTM_PARITY)
    def test_numerically_identical(self, name, has_features, wide):
        """Same weights + same input => bit-identical logits."""
        import reference_conv_lstm

        kwargs = self._kwargs(has_features, wide)
        reference = getattr(reference_conv_lstm, name)(**kwargs)
        built = get_model(name, **kwargs)
        built.load_state_dict(reference.state_dict())
        reference.eval()
        built.eval()

        batch = 3
        signal = torch.randn(batch, self.BASE["signal_len"])
        sequence = torch.randn(batch, 4, self.BASE["kmer_len"])
        feat_len = self.BASE["kmer_len"] + (2 * self.DWELL_MARGIN if wide else 0)
        features = torch.randn(batch, 5, feat_len)

        with torch.no_grad():
            if has_features:
                expected = reference(signal, sequence, features)
                actual = built(signal, sequence, features)
            else:
                expected = reference(signal, sequence)
                actual = built(signal, sequence)
        assert torch.equal(expected, actual)

    def test_same_seed_initialisation_matches(self):
        """Module construction order is preserved, so seeded init is unchanged."""
        import reference_conv_lstm

        kwargs = self._kwargs(True, True)
        torch.manual_seed(1234)
        reference = reference_conv_lstm.ConvLSTMDwellGNAttn(**kwargs)
        torch.manual_seed(1234)
        built = get_model("ConvLSTMDwellGNAttn", **kwargs)
        ref_sd, built_sd = reference.state_dict(), built.state_dict()
        assert list(ref_sd) == list(built_sd)
        for key in ref_sd:
            assert torch.equal(ref_sd[key], built_sd[key]), key

    def test_signal_kmer_encoding_parity(self):
        """The optional seq_pool node must appear only for signal_kmer input."""
        import reference_conv_lstm

        kwargs = dict(self.BASE, num_features=5, seq_encoding="signal_kmer")
        reference = reference_conv_lstm.ConvLSTMDwell(**kwargs)
        built = get_model("ConvLSTMDwell", **kwargs)
        built.load_state_dict(reference.state_dict())
        reference.eval()
        built.eval()

        signal = torch.randn(2, 100)
        sequence = torch.randn(2, 36, 100)
        features = torch.randn(2, 5, 11)
        with torch.no_grad():
            assert torch.equal(
                reference(signal, sequence, features), built(signal, sequence, features)
            )

    def test_isinstance_and_signature_preserved(self):
        """model_loading._instantiate_model relies on both."""
        import inspect

        model = get_model("ConvLSTMDwell", **self._kwargs(True, False))
        assert isinstance(model, ConvLSTMDwell)

        sig = inspect.signature(ConvLSTMDwell)
        assert not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        assert {"signal_len", "kmer_len", "num_features", "conv_channels"} <= set(sig.parameters)

    def test_unknown_kwarg_rejected(self):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            get_model("ConvLSTMDwell", not_a_real_param=1)

    def test_legacy_checkpoint_still_loads(self, tmp_path):
        """A checkpoint written by the OLD hand-written class must still load."""
        import json

        import reference_conv_lstm

        from leech.model_loading import load_model_from_checkpoint

        arch = {
            "model_name": "ConvLSTMDwellGNAttn",
            **self._kwargs(True, True),
        }
        reference = reference_conv_lstm.ConvLSTMDwellGNAttn(**self._kwargs(True, True))
        torch.save({"model_state_dict": reference.state_dict()}, tmp_path / "model_best.pt")
        # config.json as written by training, i.e. with training params mixed in
        (tmp_path / "config.json").write_text(
            json.dumps({**arch, "epochs": 3, "batch_size": 8, "device": "cpu", "seed": 1})
        )

        loaded, _ = load_model_from_checkpoint(tmp_path, device="cpu")
        reference.eval()
        loaded.eval()
        signal = torch.randn(2, self.BASE["signal_len"])
        sequence = torch.randn(2, 4, self.BASE["kmer_len"])
        features = torch.randn(2, 5, self.BASE["kmer_len"] + 2 * self.DWELL_MARGIN)
        with torch.no_grad():
            assert torch.equal(
                reference(signal, sequence, features), loaded(signal, sequence, features)
            )


# Config-built name -> (reference class name, norm_type, signal channels)
_TCN_PARITY = [
    ("TCNDwell", "TCNDwell", "batchnorm", 1),
    ("TCNDwellGN", "TCNDwell", "groupnorm", 1),
    ("TCNDwellLN", "TCNDwell", "layernorm", 1),
    ("TCNDwellResidual", "TCNDwellResidual", "batchnorm", 2),
    ("TCNDwellResidualGN", "TCNDwellResidual", "groupnorm", 2),
    ("TCNDwellResidualLN", "TCNDwellResidual", "layernorm", 2),
    ("TCNDwellSplitResidual", "TCNDwellSplitResidual", "batchnorm", 2),
    ("TCNDwellSplitResidualLN", "TCNDwellSplitResidual", "layernorm", 2),
    ("TCNDwellResidualMotor", "TCNDwellResidualMotor", "batchnorm", 2),
    ("TCNDwellResidualLNMotor", "TCNDwellResidualMotor", "layernorm", 2),
    ("TCNDwellResidualDwellAttn", "TCNDwellResidualDwellAttn", "batchnorm", 2),
    ("TCNDwellResidualLNDwellAttn", "TCNDwellResidualDwellAttn", "layernorm", 2),
]

# The Motor classes rebuilt `self.classifier` after ``super().__init__()`` had
# already built one; the discarded head consumed random numbers that the graph
# does not. Key names, key order and forward outputs still match exactly — only
# the RNG stream differs, and these names were never registered before the
# config layer existed, so no checkpoint depends on it.
_TCN_SEEDED_INIT_EXCEPTIONS = {"TCNDwellResidualMotor", "TCNDwellResidualLNMotor"}


class TestTCNConfigDrivenParity:
    """Config-built TCN models must match the classes they replaced."""

    DWELL_MARGIN = 5
    BASE = {
        "signal_len": 100,
        "kmer_len": 11,
        "num_features": 5,
        "hidden_channels": 16,
        "num_layers": 2,
        "dwell_margin": DWELL_MARGIN,
    }

    def _kwargs(self, norm_type):
        return dict(self.BASE, norm_type=norm_type)

    def _inputs(self, channels, batch=3):
        signal = torch.randn(batch, channels, 100) if channels > 1 else torch.randn(batch, 100)
        sequence = torch.randn(batch, 4, self.BASE["kmer_len"])
        features = torch.randn(batch, 5, self.BASE["kmer_len"] + 2 * self.DWELL_MARGIN)
        return signal, sequence, features

    @pytest.mark.parametrize("name,ref_name,norm_type,channels", _TCN_PARITY)
    def test_state_dict_keys_unchanged(self, name, ref_name, norm_type, channels):
        """Checkpoint compatibility: parameter names and their order must not drift."""
        import reference_tcn

        kwargs = self._kwargs(norm_type)
        reference = getattr(reference_tcn, ref_name)(**kwargs)
        built = get_model(name, **kwargs)
        assert list(built.state_dict()) == list(reference.state_dict())

    @pytest.mark.parametrize("name,ref_name,norm_type,channels", _TCN_PARITY)
    def test_numerically_identical(self, name, ref_name, norm_type, channels):
        """Same weights + same input => bit-identical logits."""
        import reference_tcn

        kwargs = self._kwargs(norm_type)
        reference = getattr(reference_tcn, ref_name)(**kwargs)
        built = get_model(name, **kwargs)
        built.load_state_dict(reference.state_dict())
        reference.eval()
        built.eval()

        signal, sequence, features = self._inputs(channels)
        with torch.no_grad():
            assert torch.equal(
                reference(signal, sequence, features), built(signal, sequence, features)
            )

    @pytest.mark.parametrize("name,ref_name,norm_type,channels", _TCN_PARITY)
    def test_same_seed_initialisation_matches(self, name, ref_name, norm_type, channels):
        """Module construction order is preserved, so seeded init is unchanged."""
        import reference_tcn

        if name in _TCN_SEEDED_INIT_EXCEPTIONS:
            pytest.skip("old class discarded a classifier head, perturbing the RNG stream")

        kwargs = self._kwargs(norm_type)
        torch.manual_seed(1234)
        reference = getattr(reference_tcn, ref_name)(**kwargs)
        torch.manual_seed(1234)
        built = get_model(name, **kwargs)
        ref_sd, built_sd = reference.state_dict(), built.state_dict()
        assert list(ref_sd) == list(built_sd)
        for key in ref_sd:
            assert torch.equal(ref_sd[key], built_sd[key]), key

    def test_signal_kmer_encoding_parity(self):
        """The optional seq_pool node must appear only for signal_kmer input."""
        import reference_tcn

        kwargs = dict(self.BASE, seq_encoding="signal_kmer")
        reference = reference_tcn.TCNDwell(**kwargs)
        built = get_model("TCNDwell", **kwargs)
        built.load_state_dict(reference.state_dict())
        assert list(built.state_dict()) == list(reference.state_dict())
        reference.eval()
        built.eval()

        signal = torch.randn(2, 100)
        sequence = torch.randn(2, 36, 100)
        features = torch.randn(2, 5, self.BASE["kmer_len"] + 2 * self.DWELL_MARGIN)
        with torch.no_grad():
            assert torch.equal(
                reference(signal, sequence, features), built(signal, sequence, features)
            )

    def test_split_residual_single_channel_fallback(self):
        """A 1-channel signal must still feed both split branches, as before."""
        import reference_tcn

        kwargs = self._kwargs("layernorm")
        reference = reference_tcn.TCNDwellSplitResidual(**kwargs)
        built = get_model("TCNDwellSplitResidualLN", **kwargs)
        built.load_state_dict(reference.state_dict())
        reference.eval()
        built.eval()

        signal, sequence, features = self._inputs(1)
        with torch.no_grad():
            assert torch.equal(
                reference(signal, sequence, features), built(signal, sequence, features)
            )

    def test_isinstance_and_signature_preserved(self):
        """model_loading._instantiate_model relies on the explicit signature."""
        import inspect

        from leech.models import nn as leech_nn

        model = get_model("TCNDwellGN", **self.BASE)
        assert isinstance(model, leech_nn.Graph)
        assert type(model).__name__ == "TCNDwellGN"

        sig = inspect.signature(TCNDwell)
        assert not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        assert {"signal_len", "kmer_len", "num_features", "hidden_channels", "norm_type"} <= set(
            sig.parameters
        )

    def test_motor_signature_exposes_motor_params(self):
        """The old **kwargs constructor made _instantiate_model drop these."""
        import inspect

        sig = inspect.signature(MODEL_REGISTRY["TCNDwellResidualMotor"])
        assert {"motor_pool_start", "motor_pool_end", "motor_proj_dim"} <= set(sig.parameters)

        sig = inspect.signature(MODEL_REGISTRY["TCNDwellResidualDwellAttn"])
        assert {"num_dwell_features", "dwell_conv_channels", "num_dwell_attn_heads"} <= set(
            sig.parameters
        )

    def test_legacy_checkpoint_still_loads(self, tmp_path):
        """A checkpoint written by the OLD hand-written class must still load."""
        import json

        import reference_tcn

        from leech.model_loading import load_model_from_checkpoint

        kwargs = self._kwargs("layernorm")
        arch = {"model_name": "TCNDwellResidualLN", **kwargs}
        reference = reference_tcn.TCNDwellResidual(**kwargs)
        torch.save({"model_state_dict": reference.state_dict()}, tmp_path / "model_best.pt")
        # config.json as written by training, i.e. with training params mixed in
        (tmp_path / "config.json").write_text(
            json.dumps({**arch, "epochs": 3, "batch_size": 8, "device": "cpu", "seed": 1})
        )

        loaded, _ = load_model_from_checkpoint(tmp_path, device="cpu")
        reference.eval()
        loaded.eval()
        signal, sequence, features = self._inputs(2, batch=2)
        with torch.no_grad():
            assert torch.equal(
                reference(signal, sequence, features), loaded(signal, sequence, features)
            )


class TestRegistryCompleteness:
    """Every registered name must be constructible and exported."""

    @pytest.mark.parametrize("model_name", sorted(MODEL_REGISTRY.keys()))
    def test_every_registered_name_constructs(self, model_name):
        model = get_model(model_name, **small_model_kwargs(model_name))
        assert sum(p.numel() for p in model.parameters()) > 0
        assert type(model).__name__ == model_name

    def test_previously_orphaned_models_are_registered(self):
        """These appeared in FEATURE_MODELS but not in the registry (issue fixed)."""
        from leech.models.inference_wrapper import ModelInferenceWrapper

        orphans = {
            "TCNDwellResidualMotor",
            "TCNDwellResidualLNMotor",
            "TCNDwellResidualDwellAttn",
            "TCNDwellResidualLNDwellAttn",
        }
        assert orphans <= set(MODEL_REGISTRY.keys())
        # and the inference path's model sets stay consistent with the registry
        assert ModelInferenceWrapper.FEATURE_MODELS <= set(MODEL_REGISTRY.keys())
        assert ModelInferenceWrapper.WIDE_FEATURE_MODELS <= set(MODEL_REGISTRY.keys())

    def test_package_attribute_access(self):
        """`from leech.models import <Name>` works for config-declared names."""
        import leech.models as models_pkg

        for name in MODEL_REGISTRY.keys():
            assert getattr(models_pkg, name) is MODEL_REGISTRY[name]

    def test_config_discovery_is_torch_free(self):
        """Parsing TOML configs must not drag torch into the CLI's import path."""
        import subprocess
        import sys

        code = (
            "import sys; from leech.models.config_loader import discover_configs; "
            "assert 'ConvLSTMDwell' in discover_configs(); "
            "assert 'torch' not in sys.modules, 'config discovery imported torch'"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
