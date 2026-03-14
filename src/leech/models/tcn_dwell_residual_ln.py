"""
TCNDwellResidualLN: TCNDwellResidual with LayerNorm.

Same architecture as TCNDwellResidual (2-channel signal + cross-attention)
but uses LayerNorm instead of BatchNorm in all temporal blocks, feature
branch, and classifier MLP.
"""

from leech.models.tcn_dwell_residual import TCNDwellResidual


class TCNDwellResidualLN(TCNDwellResidual):
    """
    TCNDwellResidual with LayerNorm instead of BatchNorm.

    Inherits all architecture from TCNDwellResidual, overriding only the
    default norm_type to "layernorm".

    All constructor arguments are identical to TCNDwellResidual.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("norm_type", "layernorm")
        super().__init__(**kwargs)
