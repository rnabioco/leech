"""
Tests for model architectures.

Tests all 6 model architectures: ConvLSTMBase, ConvLSTMDwell,
TransformerDwell, ConvOnly, TCNDwell, ResNetDwell.
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


class TestModelRegistry:
    """Test model registry and get_model function."""

    def test_model_registry_complete(self):
        """Test that MODEL_REGISTRY contains all expected models."""
        expected_models = {
            "ConvLSTMBase",
            "ConvLSTMDwell",
            "TransformerDwell",
            "ConvOnly",
            "TCNDwell",
            "ResNetDwell",
        }
        assert set(MODEL_REGISTRY.keys()) == expected_models

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
        """Test forward pass with valid inputs."""
        model = TransformerDwell(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            num_features=model_config["num_features"],
            d_model=64,
            nhead=4,
            num_layers=2,
        )
        model.eval()

        batch_size = 4
        signal = torch.randn(batch_size, model_config["signal_len"])
        sequence = torch.randn(batch_size, 4, model_config["kmer_len"])
        features = torch.randn(batch_size, model_config["num_features"], model_config["kmer_len"])

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
        """Test forward pass with valid inputs."""
        model = ConvOnly(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            num_features=model_config["num_features"],
            base_channels=8,
            num_blocks=2,
        )
        model.eval()

        batch_size = 4
        signal = torch.randn(batch_size, model_config["signal_len"])
        sequence = torch.randn(batch_size, 4, model_config["kmer_len"])
        features = torch.randn(batch_size, model_config["num_features"], model_config["kmer_len"])

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
        assert len(model.feature_conv) == 2


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
        """Test forward pass with valid inputs."""
        model = TCNDwell(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            num_features=model_config["num_features"],
            hidden_channels=16,
            num_layers=4,
            kernel_size=3,
        )
        model.eval()

        batch_size = 4
        signal = torch.randn(batch_size, model_config["signal_len"])
        sequence = torch.randn(batch_size, 4, model_config["kmer_len"])
        features = torch.randn(batch_size, model_config["num_features"], model_config["kmer_len"])

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
        """Test forward pass with valid inputs."""
        model = ResNetDwell(
            signal_len=model_config["signal_len"],
            kmer_len=model_config["kmer_len"],
            num_features=model_config["num_features"],
            base_channels=8,
        )
        model.eval()

        batch_size = 4
        signal = torch.randn(batch_size, model_config["signal_len"])
        sequence = torch.randn(batch_size, 4, model_config["kmer_len"])
        features = torch.randn(batch_size, model_config["num_features"], model_config["kmer_len"])

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
        assert hasattr(model, "feature_resnet")


class TestModelComparisons:
    """Test comparisons across all models."""

    @pytest.mark.parametrize(
        "model_name,requires_features",
        [
            ("ConvLSTMBase", False),
            ("ConvLSTMDwell", True),
            ("TransformerDwell", True),
            ("ConvOnly", True),
            ("TCNDwell", True),
            ("ResNetDwell", True),
        ],
    )
    def test_all_models_forward(self, model_name, requires_features):
        """Test that all models can do forward pass."""
        # Base config for all models
        config = {
            "signal_len": 100,
            "kmer_len": 11,
        }

        # Model-specific configs
        if model_name == "ConvLSTMBase":
            config.update({"conv_channels": [4, 16, 32], "lstm_hidden": 16})
        elif model_name == "ConvLSTMDwell":
            config.update({"num_features": 5, "conv_channels": [4, 16, 32], "lstm_hidden": 16})
        elif model_name == "TransformerDwell":
            config.update({"num_features": 5, "d_model": 32, "nhead": 4, "num_layers": 1})
        elif model_name == "ConvOnly":
            config.update({"num_features": 5, "base_channels": 4})
        elif model_name == "ResNetDwell":
            config.update({"num_features": 5, "base_channels": 4})
        elif model_name == "TCNDwell":
            config.update(
                {"num_features": 5, "hidden_channels": 16, "num_layers": 2, "kernel_size": 3}
            )

        model = get_model(model_name, **config)
        model.eval()

        batch_size = 2
        signal = torch.randn(batch_size, config["signal_len"])
        sequence = torch.randn(batch_size, 4, config["kmer_len"])
        if requires_features:
            features = torch.randn(batch_size, config["num_features"], config["kmer_len"])
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

    def test_model_parameter_counts(self):
        """Test that models have reasonable parameter counts."""
        base_config = {
            "signal_len": 400,
            "kmer_len": 11,
        }

        for model_name in MODEL_REGISTRY.keys():
            # Model-specific configs
            if model_name == "ConvLSTMBase":
                model_config = {**base_config, "conv_channels": [4, 16, 64], "lstm_hidden": 32}
            elif model_name == "ConvLSTMDwell":
                model_config = {
                    **base_config,
                    "num_features": 5,
                    "conv_channels": [4, 16, 64],
                    "lstm_hidden": 32,
                }
            elif model_name == "TransformerDwell":
                model_config = {
                    **base_config,
                    "num_features": 5,
                    "d_model": 64,
                    "nhead": 4,
                    "num_layers": 2,
                }
            elif model_name == "ConvOnly":
                model_config = {**base_config, "num_features": 5, "base_channels": 8}
            elif model_name == "ResNetDwell":
                model_config = {**base_config, "num_features": 5, "base_channels": 8}
            elif model_name == "TCNDwell":
                model_config = {
                    **base_config,
                    "num_features": 5,
                    "hidden_channels": 32,
                    "num_layers": 3,
                    "kernel_size": 3,
                }

            model = get_model(model_name, **model_config)
            param_count = sum(p.numel() for p in model.parameters())

            # Models should have between 1k and 10M parameters
            assert 1_000 < param_count < 10_000_000, f"{model_name} has {param_count} parameters"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
