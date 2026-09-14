"""
Reusable neural network components for leech models.

This module provides shared building blocks used across all model architectures,
eliminating code duplication and making it easier to create new models.
"""

import functools

import numpy as np
import torch
import torch.nn as nn

from leech.constants import (
    DEFAULT_CAUSAL_TCN,
    DEFAULT_CONV_CHANNELS,
    DEFAULT_DROPOUT,
    DEFAULT_FEATURE_KERNEL,
    DEFAULT_SEQ_KERNEL,
    DEFAULT_SIGNAL_KERNEL,
    FEATURE_STANDARDIZE_EPS,
)


@functools.lru_cache(maxsize=128)
def _segment_mean_weights(length: int, output_size: int) -> np.ndarray:
    """The cached half of :func:`segment_mean_matrix`, as **numpy**.

    Deliberately not a ``torch.Tensor``. Under ``torch.export``'s fake-tensor
    tracing, ``torch.from_numpy`` returns a *FakeTensor*; cached, that fake
    outlives the trace and poisons every later eager call — the model then
    returns a tensor subclass and ``verify_onnx``'s ``.numpy()`` dies a long
    way from the cause. Cache the array, build the tensor per call: the build
    is a 4 KB copy and the bin arithmetic is what was worth caching.

    Built in float64 so the reciprocals are exact before they are rounded once
    into the model's dtype.

    ``output_size > length`` is legal and is not a mistake: ``ResNetDwell``
    pools a length-4 feature map up to ``kmer_len`` 11, and ``adaptive_avg_pool``
    UPSAMPLES there by repeating bins. The bin formula covers it unchanged —
    ``ceil((j+1)*L/K) > floor(j*L/K)`` for every ``j`` whenever ``L >= 1``, so no
    bin is ever empty. An earlier version of this rejected it as out of range
    and two model tests caught it.
    """
    if length < 1 or output_size < 1:
        raise ValueError(f"length {length} and output_size {output_size} must both be >= 1")
    w = np.zeros((length, output_size), dtype=np.float64)
    for j in range(output_size):
        start = (j * length) // output_size
        end = -((-(j + 1) * length) // output_size)  # ceil((j+1)*L/K)
        w[start:end, j] = 1.0 / (end - start)
    return w


def segment_mean_matrix(
    length: int,
    output_size: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """``[length, output_size]``: column *j* averages ``x[start_j:end_j]``.

    PyTorch's adaptive-pool bin rule, written out: bin *j* covers
    ``[floor(j*L/K), ceil((j+1)*L/K))``, so the bins tile the axis and their
    widths differ by at most one. Right-multiplying by this matrix *is*
    ``adaptive_avg_pool1d`` — see :class:`AdaptiveAvgPool1d` for why leech
    spells it that way.

    Always builds a fresh tensor (never the :func:`_cached_segment_mean_tensor`
    :class:`AdaptiveAvgPool1d.forward` uses in eager mode) — this is the path
    tracing/export takes, where a cached tensor is a poisoned FakeTensor
    waiting to leak into a later real call.
    """
    return torch.tensor(
        _segment_mean_weights(int(length), int(output_size)), dtype=dtype, device=device
    )


@functools.lru_cache(maxsize=128)
def _cached_segment_mean_tensor(
    length: int, output_size: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """The torch.Tensor :func:`segment_mean_matrix` builds, cached this time.

    Every eager forward through :class:`AdaptiveAvgPool1d` at a given
    ``(length, output_size, dtype, device)`` used to pay a fresh
    ``torch.tensor(...)`` construction — an H2D copy on every pool, every
    step, of a 4 KB constant that never changes for a fixed model and input
    length. Safe to cache *here*, unlike the plain tensor form: this function
    is only ever called from :meth:`AdaptiveAvgPool1d.forward` when
    ``torch.compiler.is_compiling()`` is false, so nothing that reaches this
    cache is a FakeTensor built under tracing (see :func:`segment_mean_matrix`
    for the path tracing/export takes instead, which never touches this
    cache).

    Built inside ``torch.inference_mode(False)`` regardless of the caller's
    own mode. Without that, the first eager call at a given key made from
    inside ``validate()``/``eval test``/``predict``/``bundle`` (all run
    under ``torch.inference_mode()``) would cache an *inference tensor* —
    and this cache is process-global and outlives any one call, so a later
    grad-enabled forward at that same ``(length, output_size)`` (same
    process, e.g. a pytest session running eval tests before training ones,
    or a script that evaluates then fine-tunes) reuses it and dies in
    ``backward()`` with "Inference tensors cannot be saved for backward" —
    the exact FakeTensor-style poisoning risk the class-level docstring
    above already warns about, just from ``inference_mode`` instead of
    export tracing. Escaping inference_mode for the build makes the cached
    tensor safe to read from *either* context afterwards.
    """
    with torch.inference_mode(False):
        return torch.tensor(_segment_mean_weights(length, output_size), dtype=dtype, device=device)


class AdaptiveAvgPool1d(nn.AdaptiveAvgPool1d):
    """``nn.AdaptiveAvgPool1d`` written as one matmul against a constant.

    Same arithmetic, same ``output_size``, no parameters — and an ONNX graph a
    non-Python runtime can actually load. ``aten::adaptive_avg_pool1d`` has no
    ONNX op behind it when the output size does not divide the input size, so
    every exporter has to open-code it:

    * the **legacy TorchScript** exporter refuses outright
      (``SymbolicValueError: ... output size that are not factor of input
      size``);
    * the **dynamo** exporter open-codes it as
      ``Unsqueeze -> Transpose -> GatherND -> Transpose -> Where(masked_fill)``
      followed by one ``Gather``+``Add`` per element of the widest bin. That is
      a rank-8 gather over an all-constant index and mask, and tract 0.23.5
      gives up on it during shape analysis — which is what made
      ``charging_tcn_rna004@v0.1.0`` unloadable by the shipped ``escpod``
      binaries (rnabioco/escapepod-models#96). No graph rewrite fixes it
      afterwards: onnx-simplifier folds away every ``Shape`` node and still
      leaves the ``GatherND``, and onnxruntime's optimiser keeps it and adds
      ORT-only fusions on top.

    Written out as a matmul, the same two pools become one ``MatMul`` each
    against a ``[L_in, L_out]`` initializer, and the graph loads.

    The matmul is forced to float32 outside autocast: ``adaptive_avg_pool1d``
    is not on autocast's cast list, so it keeps its input dtype and accumulates
    in float32, and a bare ``@`` under autocast would silently become an fp16
    gemm. Keeping the accumulation in float32 is what makes this a rewrite of
    the same function rather than a change to it.
    """

    def _output_size(self) -> int:
        size = self.output_size
        if isinstance(size, tuple | list):
            (size,) = size
        return int(size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        if not isinstance(length, int):
            # The pooled axis is not a concrete length, so the bin edges are
            # not knowable and there is no constant matrix to build. Fall back
            # to the aten op — which is the graph this class exists to avoid,
            # so it is worth knowing exactly when this fires:
            #
            #   eager                     never (a shape is an int)
            #   torch.export / dynamo     only if the LAST axis is declared
            #                             dynamic; leech's exports make only
            #                             the batch axis dynamic, and the
            #                             charging TCN's graph is 8 MatMuls
            #                             and zero GatherND, so it does not
            #   torch.compile, dynamic    yes, and correctly — an unknown
            #                             length has no constant matrix
            #   torch.jit.trace           YES. Tracing makes `.shape[-1]` a
            #                             Tensor, so this branch is taken and
            #                             the legacy TorchScript ONNX
            #                             exporter still meets an
            #                             `adaptive_avg_pool1d` it refuses.
            #                             Measured, not assumed; see
            #                             `leech.onnx_export`'s docstring.
            return nn.functional.adaptive_avg_pool1d(x, self._output_size())
        output_size = self._output_size()
        if torch.compiler.is_compiling():
            # Tracing (torch.compile or torch.export): never read or write
            # the eager cache below, which would leak a FakeTensor built
            # under this trace into a later real forward call.
            w = segment_mean_matrix(length, output_size, dtype=torch.float32, device=x.device)
        else:
            w = _cached_segment_mean_tensor(length, output_size, torch.float32, x.device)
        with torch.amp.autocast(x.device.type, enabled=False):
            out = torch.matmul(x.float(), w)
        return out.to(x.dtype)


def make_norm(norm_type: str, num_channels: int) -> nn.Module:
    """Create a normalization layer.

    Args:
        norm_type: One of "batchnorm", "groupnorm", "layernorm".
        num_channels: Number of channels to normalize.

    Returns:
        Normalization module.
    """
    if norm_type == "batchnorm":
        return nn.BatchNorm1d(num_channels)
    elif norm_type == "groupnorm":
        num_groups = min(4, num_channels)
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif norm_type == "layernorm":
        return nn.GroupNorm(num_groups=1, num_channels=num_channels)
    else:
        raise ValueError(
            f"Unknown norm_type '{norm_type}'. Use 'batchnorm', 'groupnorm', or 'layernorm'."
        )


def _resolve_norm_type(norm_type: str, use_batchnorm: bool) -> str | None:
    """Resolve norm_type from either the new norm_type or legacy use_batchnorm flag.

    Returns None for no normalization, or a valid norm_type string.
    """
    if norm_type != "none":
        return norm_type
    if use_batchnorm:
        return "batchnorm"
    return None


class SignalBranch(nn.Module):
    """
    Reusable 1D convolutional branch for raw signal processing.

    Applies a series of 1D convolutions to extract features from nanopore signal data.

    Args:
        in_channels: Number of input channels (1 for raw signal, 2 for signal + residual)
        conv_channels: List of channel sizes for conv layers (default: [4, 16, 256])
        kernel_size: Kernel size for convolutions (default: 5)
        use_batchnorm: Insert BatchNorm1d after each Conv1d (default: False)
        norm_type: Normalization type ("none", "batchnorm", "groupnorm", "layernorm")

    Input shape:
        (batch_size, signal_len) when in_channels=1
        (batch_size, in_channels, signal_len) when in_channels>1

    Output shape:
        (batch_size, conv_channels[-1], signal_len)
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: list[int] | None = None,
        kernel_size: int = DEFAULT_SIGNAL_KERNEL,
        use_batchnorm: bool = False,
        norm_type: str = "none",
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = DEFAULT_CONV_CHANNELS

        self.in_channels = in_channels
        resolved = _resolve_norm_type(norm_type, use_batchnorm)

        layers: list[nn.Module] = []
        in_ch = in_channels
        for out_ch in conv_channels:
            layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
            )
            if resolved is not None:
                layers.append(make_norm(resolved, out_ch))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through signal branch.

        Args:
            signal: Raw signal tensor (batch_size, signal_len) for 1-channel,
                or (batch_size, in_channels, signal_len) for multi-channel

        Returns:
            Extracted features (batch_size, conv_channels[-1], signal_len)
        """
        # Add channel dimension only for single-channel input
        if signal.dim() == 2:
            signal = signal.unsqueeze(1)
        output: torch.Tensor = self.conv_layers(signal)
        return output


class SequenceBranch(nn.Module):
    """
    Reusable 1D convolutional branch for sequence processing.

    Applies a series of 1D convolutions to extract features from encoded sequences.
    Supports both base_onehot (4 channels) and signal_kmer (4*kmer_len channels).

    Args:
        in_channels: Number of input channels (4 for base_onehot, 36 for signal_kmer with (4,4) context)
        conv_channels: List of channel sizes for conv layers (default: [4, 16, 256])
        kernel_size: Kernel size for convolutions (default: 3)
        use_batchnorm: Insert BatchNorm1d after each Conv1d (default: False)
        norm_type: Normalization type ("none", "batchnorm", "groupnorm", "layernorm")

    Input shape:
        (batch_size, in_channels, seq_len)

    Output shape:
        (batch_size, conv_channels[-1], seq_len)
    """

    def __init__(
        self,
        in_channels: int = 4,
        conv_channels: list[int] | None = None,
        kernel_size: int = DEFAULT_SEQ_KERNEL,
        use_batchnorm: bool = False,
        norm_type: str = "none",
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = DEFAULT_CONV_CHANNELS

        resolved = _resolve_norm_type(norm_type, use_batchnorm)

        layers: list[nn.Module] = []
        in_ch = in_channels
        for out_ch in conv_channels:
            layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
            )
            if resolved is not None:
                layers.append(make_norm(resolved, out_ch))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through sequence branch.

        Args:
            sequence: One-hot encoded sequence (batch_size, 4, kmer_len)

        Returns:
            Extracted features (batch_size, conv_channels[-1], kmer_len)
        """
        output: torch.Tensor = self.conv_layers(sequence)
        return output


class AffineStandardize(nn.Module):
    """Frozen per-channel affine standardization: ``(x - mean) / std``.

    ``mean``/``std`` are corpus-wide per-channel statistics computed once
    over the training corpus (``LeechDataset``'s ``--standardize-features``
    pass -- see ``leech.dataset.LeechDataset.__init__``) and registered as
    buffers rather than parameters: they ride along in every checkpoint and
    every exported ONNX graph (a plain ``Sub``/``Div`` against two constant
    initializers), but are never touched by an optimizer and carry no
    gradient of their own to save.

    Broadcasts over the window axis: a ``(num_features,)`` vector applies to
    a ``(batch, num_features, kmer_len)`` input, one scale per feature
    channel regardless of window width. Channel order must match
    ``leech.chunking.extractor.merge_feature_channels`` (dwell rows, then
    signal-level rows, then k-mer residual rows) -- that is the order every
    feature array already has, and the only order this class knows about.
    """

    def __init__(self, mean: list[float], std: list[float]) -> None:
        super().__init__()
        mean_t = torch.as_tensor(mean, dtype=torch.float32)
        std_t = torch.as_tensor(std, dtype=torch.float32)
        if mean_t.shape != std_t.shape:
            raise ValueError(
                f"feature_mean (len {mean_t.numel()}) and feature_std "
                f"(len {std_t.numel()}) must have the same length"
            )
        # Floor the std so a near-constant channel (dwell_ratio sits close to
        # 1 in some corpora) divides by something other than ~0 instead of
        # producing inf/nan for every chunk. clamp_min alone would not rescue
        # a NaN std -- torch.std's unbiased (N-1) estimator gives 0/0 = NaN
        # when a channel's total sample count is 1 (a single-chunk corpus, or
        # kmer_len=1), and clamp_min(NaN, eps) is still NaN under IEEE754 --
        # so replace NaN with "no scaling" (std=1) before flooring.
        std_t = torch.nan_to_num(std_t, nan=1.0).clamp_min(FEATURE_STANDARDIZE_EPS)
        self.register_buffer("mean", mean_t.view(1, -1, 1))
        self.register_buffer("std", std_t.view(1, -1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.mean, torch.Tensor)
        assert isinstance(self.std, torch.Tensor)
        return (x - self.mean) / self.std


class FeatureBranch(nn.Module):
    """
    Reusable 1D convolutional branch for engineered features (dwell + signal levels).

    Applies a series of 1D convolutions to extract patterns from feature channels.

    Args:
        num_features: Number of input feature channels (e.g., 5 for dwell + 4 signal stats)
        conv_channels: List of channel sizes for conv layers (default: [4, 16, 256])
        kernel_size: Kernel size for convolutions (default: 3)
        use_batchnorm: Insert BatchNorm1d after each Conv1d (default: False)
        norm_type: Normalization type ("none", "batchnorm", "groupnorm", "layernorm")
        feature_mean: Optional per-channel corpus mean (length ``num_features``).
            When given together with ``feature_std``, a frozen
            :class:`AffineStandardize` layer is prepended before the first
            Conv1d, so raw feature scales (a dwell sample count, a MAD-unit
            level stat, ...) never reach a conv weight unscaled. Both or
            neither -- see ``--standardize-features``.
        feature_std: Optional per-channel corpus std (length ``num_features``).

    Input shape:
        (batch_size, num_features, kmer_len)

    Output shape:
        (batch_size, conv_channels[-1], kmer_len)
    """

    def __init__(
        self,
        num_features: int,
        conv_channels: list[int] | None = None,
        kernel_size: int = DEFAULT_FEATURE_KERNEL,
        use_batchnorm: bool = False,
        norm_type: str = "none",
        feature_mean: list[float] | None = None,
        feature_std: list[float] | None = None,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = DEFAULT_CONV_CHANNELS

        resolved = _resolve_norm_type(norm_type, use_batchnorm)

        layers: list[nn.Module] = []
        if feature_mean is not None or feature_std is not None:
            if feature_mean is None or feature_std is None:
                raise ValueError("feature_mean and feature_std must both be given or both omitted")
            if len(feature_mean) != num_features or len(feature_std) != num_features:
                raise ValueError(
                    f"feature_mean/feature_std length must equal num_features "
                    f"({num_features}); got {len(feature_mean)}/{len(feature_std)}"
                )
            layers.append(AffineStandardize(feature_mean, feature_std))
        in_ch = num_features
        for out_ch in conv_channels:
            layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
            )
            if resolved is not None:
                layers.append(make_norm(resolved, out_ch))
            layers.append(nn.ReLU())
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature branch.

        Args:
            features: Engineered features (batch_size, num_features, kmer_len)

        Returns:
            Extracted features (batch_size, conv_channels[-1], kmer_len)
        """
        output: torch.Tensor = self.conv_layers(features)
        return output


class TemporalBlock(nn.Module):
    """
    Temporal convolutional block with dilated convolutions and residual connection.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        dilation: Dilation rate
        dropout: Dropout probability
        norm_type: Normalization type ("batchnorm", "groupnorm", "layernorm")
        causal: When True (default), pad left-only so position *i* only ever
            sees positions ``<= i`` -- the sequence-modelling convention,
            appropriate when the model must not look ahead. leech's chunk
            classifiers have the whole fixed window available at every
            position, so causal padding just halves the receptive field for
            no benefit (6 layers, kernel 3 reaches 253 samples causally vs.
            127 samples on each side symmetrically). ``causal=False`` splits
            the same total padding ``(kernel_size - 1) * dilation`` evenly
            across both sides instead. Neither conv gains or loses a
            parameter either way -- only the padding changes -- so a
            checkpoint's ``state_dict()`` keys and shapes are identical
            regardless of this flag; the *values* trained under one mode are
            not a meaningful initialization for the other.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = DEFAULT_DROPOUT,
        norm_type: str = "batchnorm",
        causal: bool = DEFAULT_CAUSAL_TCN,
    ):
        super().__init__()

        self.causal = causal
        # Total padding needed to keep the sequence length fixed after each
        # conv: causal puts all of it on the left, non-causal splits it evenly.
        # Precomputed once (fixed for the module's lifetime) rather than
        # rebuilt on every forward() call, of which there are two per block.
        self.padding = (kernel_size - 1) * dilation
        if causal:
            self._pad_amounts = (self.padding, 0)
        else:
            left = self.padding // 2
            self._pad_amounts = (left, self.padding - left)

        # Two convolutional layers with normalization and dropout
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # We'll manually pad
        )
        self.norm1 = make_norm(norm_type, out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )
        self.norm2 = make_norm(norm_type, out_channels)
        self.dropout2 = nn.Dropout(dropout)

        # Residual connection
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

        self.relu = nn.ReLU()

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.pad(x, self._pad_amounts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, length)

        Returns:
            Output tensor (batch, out_channels, length)
        """
        x_padded = self._pad(x)

        # First conv block
        out = self.conv1(x_padded)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self._pad(out)

        # Second conv block
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        result: torch.Tensor = self.relu(out + res)
        return result


class TCN(nn.Module):
    """
    Temporal Convolutional Network with stacked dilated convolutions.

    Args:
        in_channels: Number of input channels
        hidden_channels: Number of channels in each layer
        num_layers: Number of temporal blocks
        kernel_size: Convolution kernel size
        dropout: Dropout probability
        norm_type: Normalization type ("batchnorm", "groupnorm", "layernorm")
        causal: Passed through to every :class:`TemporalBlock` -- see there.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = DEFAULT_DROPOUT,
        norm_type: str = "batchnorm",
        causal: bool = DEFAULT_CAUSAL_TCN,
    ):
        super().__init__()

        layers = []
        for i in range(num_layers):
            dilation = 2**i  # Exponentially increasing dilation: 1, 2, 4, 8, 16, 32
            in_ch = in_channels if i == 0 else hidden_channels
            layers.append(
                TemporalBlock(
                    in_ch,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                    norm_type=norm_type,
                    causal=causal,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, length)

        Returns:
            Output tensor (batch, hidden_channels, length)
        """
        out: torch.Tensor = self.network(x)
        return out


def logits_to_positive_prob(logits: torch.Tensor, num_out: int = 1) -> torch.Tensor:
    """Positive-class probability from raw logits, for either output convention.

    ``num_out == 2`` is a two-class ``CrossEntropyLoss`` head (Remora's
    convention): softmax and take the positive column. Anything else is
    leech's single-logit BCE convention: sigmoid. One definition shared by
    every ``predict_proba`` and by :class:`RemoraModelWrapper`, which needs
    the same conversion to translate a wrapped Remora model's native 2-class
    output into leech's single-logit convention.
    """
    if num_out == 2:
        return torch.softmax(logits, dim=-1)[:, 1:2]
    return torch.sigmoid(logits)


class BaseModel(nn.Module):
    """
    Base class for all leech models with shared predict_proba() method.

    All leech models should inherit from this class to get the standard
    predict_proba() implementation and ensure consistent interfaces.
    """

    # Whether this architecture receives the full dwell margin (no
    # dwell_offset slicing) rather than the dwell-offset-sliced window.
    # Overridden per class; for TOML/Graph architectures the equivalent lives
    # in the config's ``[params].wide_features`` instead (see
    # ``leech.models.wide_features``).
    WIDE_FEATURES: bool = False

    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        """
        Predict probability of positive class (charged tRNA).

        This method wraps the forward() pass with evaluation mode and
        sigmoid/softmax activation (see ``logits_to_positive_prob``) to
        produce probabilities in [0, 1].

        Args:
            *args: Arguments passed to forward()
            **kwargs: Keyword arguments passed to forward()

        Returns:
            Probabilities for positive class (batch_size, 1)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(*args, **kwargs)
            probs = logits_to_positive_prob(logits, getattr(self, "num_out", 1))
        return probs
