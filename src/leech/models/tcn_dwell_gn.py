"""
TCNDwellGN: Temporal Convolutional Network with GroupNorm.

Same architecture as TCNDwell but uses GroupNorm instead of BatchNorm
in all temporal blocks, feature branch, and classifier MLP.
"""

from leech.models.tcn_dwell import TCNDwell


class TCNDwellGN(TCNDwell):
    """
    TCN model with GroupNorm instead of BatchNorm.

    Inherits all architecture from TCNDwell, overriding only the default
    norm_type to "groupnorm".

    All constructor arguments are identical to TCNDwell.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("norm_type", "groupnorm")
        super().__init__(**kwargs)
