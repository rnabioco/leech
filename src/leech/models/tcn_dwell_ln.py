"""
TCNDwellLN: Temporal Convolutional Network with LayerNorm.

Same architecture as TCNDwell but uses LayerNorm (GroupNorm with num_groups=1)
instead of BatchNorm in all temporal blocks, feature branch, and classifier MLP.
"""

from leech.models.tcn_dwell import TCNDwell


class TCNDwellLN(TCNDwell):
    """
    TCN model with LayerNorm instead of BatchNorm.

    Inherits all architecture from TCNDwell, overriding only the default
    norm_type to "layernorm".

    All constructor arguments are identical to TCNDwell.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("norm_type", "layernorm")
        super().__init__(**kwargs)
