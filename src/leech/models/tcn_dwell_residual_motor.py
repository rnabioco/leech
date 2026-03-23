"""
TCNDwellResidualMotor: Motor-region explicit pooling variant.

Adds a dedicated motor-region extraction pathway that bypasses cross-attention.
The feature branch output at motor positions (+8 to +12) is mean-pooled and
projected, then concatenated with the cross-attention context vector before the
classifier. This guarantees the model sees motor-region dwell even if
cross-attention focuses on the constriction zone.
"""

import torch
from torch import nn

from leech.models.components import make_norm
from leech.models.tcn_dwell_residual import TCNDwellResidual


class TCNDwellResidualMotor(TCNDwellResidual):
    """TCNDwellResidual with explicit motor-region pooling.

    Adds a parallel pathway that mean-pools feature branch output at motor
    positions and concatenates with the attention-pooled context vector.
    """

    def __init__(
        self,
        motor_pool_start: int = 8,
        motor_pool_end: int = 13,
        motor_proj_dim: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.motor_pool_start = motor_pool_start
        self.motor_pool_end = motor_pool_end

        # Feature branch output channels (find last Conv1d in sequential)
        feat_out_dim = [
            m.out_channels
            for m in self.feature_branch.conv_layers
            if isinstance(m, nn.Conv1d)
        ][-1]  # 256

        # Project pooled motor features to a compact representation
        norm_type = kwargs.get("norm_type", "batchnorm")
        self.motor_proj = nn.Sequential(
            nn.Linear(feat_out_dim, motor_proj_dim),
            nn.ReLU(),
            make_norm(norm_type, motor_proj_dim),
        )

        # Rebuild classifier with wider input (merged_dim + motor_proj_dim)
        merged_dim = kwargs.get("hidden_channels", 64) * 2
        dropout = kwargs.get("dropout", 0.1)
        num_out = kwargs.get("num_out", 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(merged_dim + motor_proj_dim, 256),
            nn.ReLU(),
            make_norm(norm_type, 256),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            make_norm(norm_type, 64),
            nn.Dropout(dropout),
            nn.Linear(64, num_out),
        )

    def forward(
        self, signal: torch.Tensor, sequence: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        # Signal branch
        if signal.dim() == 2:
            signal_feat = signal.unsqueeze(1)
        else:
            signal_feat = signal
        signal_feat = self.signal_tcn(signal_feat)
        signal_feat = self.signal_pool(signal_feat)

        # Sequence branch
        seq_feat = self.seq_tcn(sequence)
        if self.seq_encoding == "signal_kmer":
            seq_feat = self.seq_pool(seq_feat)

        # Feature branch (full width for cross-attention AND motor pooling)
        feat_out = self.feature_branch(features)  # (batch, 256, feat_len)

        # Motor-region pooling: bypass cross-attention
        motor_start = min(self.motor_pool_start, feat_out.shape[2])
        motor_end = min(self.motor_pool_end, feat_out.shape[2])
        if motor_end > motor_start:
            motor_pooled = feat_out[:, :, motor_start:motor_end].mean(dim=2)
        else:
            motor_pooled = feat_out.mean(dim=2)
        motor_vec = self.motor_proj(motor_pooled)  # (batch, motor_proj_dim)

        # Cross-attention (existing path)
        feat_kv = feat_out.transpose(1, 2)  # (batch, feat_len, 256)
        merged = torch.cat([signal_feat, seq_feat], dim=1)
        merged = merged.transpose(1, 2)  # (batch, kmer_len, merged_dim)

        attn_out, _ = self.cross_attn(query=merged, key=feat_kv, value=feat_kv)
        combined = self.attn_norm(merged + attn_out)

        # Attention pooling
        pool_scores = self.pool_linear(combined)
        pool_weights = torch.softmax(pool_scores, dim=1)
        context_vector = (pool_weights * combined).sum(dim=1)  # (batch, merged_dim)

        # Concatenate motor pathway with attention pathway
        context_vector = torch.cat([context_vector, motor_vec], dim=1)

        logits: torch.Tensor = self.classifier(context_vector)
        return logits


class TCNDwellResidualLNMotor(TCNDwellResidualMotor):
    """TCNDwellResidualMotor with LayerNorm."""

    def __init__(self, **kwargs):
        kwargs.setdefault("norm_type", "layernorm")
        super().__init__(**kwargs)
