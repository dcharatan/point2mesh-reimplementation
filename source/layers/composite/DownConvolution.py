import tensorflow as tf
from typing import List, Optional
from tensorflow.keras.layers import Layer, LeakyReLU, BatchNormalization
from ...mesh.Mesh import Mesh
from ..feature import features_valid
from ..pooling.MeshPool import MeshPool
from .ConvolutionSequence import ConvolutionSequence


class DownConvolution(Layer):
    """A DownConvolution combines several skip-connected ConvolutionSequence
    blocks separated by nonlinearities and batch normalization. It can
    optionally include a pooling step at the end.
    """

    convolutions: List[ConvolutionSequence]
    mesh_pool: Optional[MeshPool]
    batch_normalizations: List[BatchNormalization]
    leaky_relu: LeakyReLU

    def __init__(
        self,
        out_channels: int,
        convolutions_per_sequence: int,
        num_sequences: int,
        leaky_relu_alpha: float,
        pool_target: Optional[int],
    ):
        Layer.__init__(self)
        self.convolutions = [
            ConvolutionSequence(out_channels, convolutions_per_sequence)
            for _ in range(num_sequences)
        ]
        self.leaky_relu = LeakyReLU(leaky_relu_alpha)
        self.batch_normalizations = [BatchNormalization() for _ in range(num_sequences)]
        self.mesh_pool = None if pool_target is None else MeshPool(pool_target)

    def call(self, mesh: Mesh, features: tf.Tensor) -> tf.Tensor:
        assert features_valid(mesh, features)

        # Run through the first ConvolutionSequence.
        out_features = self.convolutions[0](mesh, features)
        out_features = self.leaky_relu(out_features)
        out_features = self.batch_normalizations[0](out_features, training=True)

        # Run the remaining ConvolutionSequences with skip connections.
        in_features = out_features
        for i in range(1, len(self.convolutions)):
            # Create out_features using in_features.
            out_features = self.convolutions[i](mesh, in_features)
            out_features = self.leaky_relu(out_features)
            out_features = self.batch_normalizations[i](out_features, training=True)

            # Add the skip connections.
            out_features += in_features
            in_features = out_features

        # Optionally run pooling.
        if self.mesh_pool is not None:
            return self.mesh_pool(mesh, out_features)
        return out_features, None
