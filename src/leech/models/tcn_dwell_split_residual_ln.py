"""
TCNDwellSplitResidualLN: TCNDwellSplitResidual with LayerNorm.

Same architecture as TCNDwellSplitResidual (separate signal/residual branches
+ cross-attention) but uses LayerNorm instead of BatchNorm.
"""

from leech.models.tcn_dwell_split_residual import TCNDwellSplitResidual


class TCNDwellSplitResidualLN(TCNDwellSplitResidual):
    """
    TCNDwellSplitResidual with LayerNorm instead of BatchNorm.

    Inherits all architecture from TCNDwellSplitResidual, overriding only the
    default norm_type to "layernorm".

    All constructor arguments are identical to TCNDwellSplitResidual.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("norm_type", "layernorm")
        super().__init__(**kwargs)
