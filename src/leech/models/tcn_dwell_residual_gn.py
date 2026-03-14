"""
TCNDwellResidualGN: TCNDwellResidual with GroupNorm.

Same architecture as TCNDwellResidual (2-channel signal + cross-attention)
but uses GroupNorm instead of BatchNorm in all temporal blocks, feature
branch, and classifier MLP.
"""

from leech.models.tcn_dwell_residual import TCNDwellResidual


class TCNDwellResidualGN(TCNDwellResidual):
    """
    TCNDwellResidual with GroupNorm instead of BatchNorm.

    Inherits all architecture from TCNDwellResidual, overriding only the
    default norm_type to "groupnorm".

    All constructor arguments are identical to TCNDwellResidual.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("norm_type", "groupnorm")
        super().__init__(**kwargs)
