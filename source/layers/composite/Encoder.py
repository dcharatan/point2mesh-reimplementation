import tensorflow as tf
from typing import List, Optional
from tensorflow.keras.layers import Layer
from .DownConvolution import DownConvolution
from ...mesh.Mesh import Mesh
from ..feature import features_valid


class Encoder(Layer):
    """This combines several DownConvolution layers."""

    down_convolutions: List[DownConvolution]

    def __init__(
        self,
        out_channels: List[int],
        convolutions_per_sequence: int,
        num_sequences_per_down_convolution: int,
        leaky_relu_alpha: float,
        pool_targets: List[Optional[int]],
    ):
        Layer.__init__(self)

        assert len(out_channels) == len(pool_targets)
        for i in range(1, len(pool_targets)):
            assert pool_targets[i] is None or pool_targets[i] < pool_targets[i - 1]

        self.down_convolutions = [
            DownConvolution(
                out_channels[i],
                convolutions_per_sequence,
                num_sequences_per_down_convolution,
                leaky_relu_alpha,
                pool_targets[i],
            )
            for i in range(len(pool_targets))
        ]

    def call(self, mesh: Mesh, features: tf.Tensor) -> tf.Tensor:
        assert features_valid(mesh, features)

        snapshots = []
        for down_convolution in self.down_convolutions:
            features, snapshot = down_convolution(mesh, features)
            snapshots.append(snapshot)

        return features, snapshots
