import tensorflow as tf
from typing import List, Optional
from tensorflow.keras.layers import Layer
from .UpConvolution import UpConvolution
from ...mesh.Mesh import Mesh
from ..feature import features_valid
from ..pooling.CollapseSnapshot import CollapseSnapshot


class Decoder(Layer):
    """This combines several UpConvolution layers."""

    up_convolutions: List[UpConvolution]

    def __init__(
        self,
        out_channels: List[int],
        convolutions_per_sequence: int,
        num_sequences_per_down_convolution: int,
        leaky_relu_alpha: float,
        encoder_pool_targets: List[Optional[int]],
    ):
        Layer.__init__(self)

        assert len(out_channels) == len(encoder_pool_targets)

        needs_unpool = [x != None for x in reversed(encoder_pool_targets)]
        self.up_convolutions = [
            UpConvolution(
                out_channels[i],
                convolutions_per_sequence,
                num_sequences_per_down_convolution,
                leaky_relu_alpha,
                needs_unpool[i],
            )
            for i in range(len(encoder_pool_targets))
        ]

    def call(
        self, mesh: Mesh, features: tf.Tensor, snapshots: CollapseSnapshot
    ) -> tf.Tensor:
        assert features_valid(mesh, features)

        snapshots = reversed(snapshots)
        for up_convolution, snapshot in zip(self.up_convolutions, snapshots):
            features = up_convolution(mesh, features, snapshot)

        return features
