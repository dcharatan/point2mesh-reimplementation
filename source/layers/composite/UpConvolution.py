import tensorflow as tf
from typing import List, Optional
from tensorflow.keras.layers import Layer, LeakyReLU, BatchNormalization
from ...mesh.Mesh import Mesh
from ..feature import features_valid
from ..pooling.MeshUnpool import MeshUnpool
from ..pooling.CollapseSnapshot import CollapseSnapshot
from .ConvolutionSequence import ConvolutionSequence


class UpConvolution(Layer):
    """An UpConvolution combines several skip-connected ConvolutionSequence
    blocks separated by nonlinearities and batch normalization. It can
    optionally include an unpooling step at the end.
    """

    convolutions: List[ConvolutionSequence]
    mesh_unpool: Optional[MeshUnpool]
    batch_normalizations: List[BatchNormalization]
    leaky_relu: LeakyReLU

    def __init__(
        self,
        out_channels: int,
        convolutions_per_sequence: int,
        num_sequences: int,
        leaky_relu_alpha: float,
        unpool: bool,
    ):
        Layer.__init__(self)
        self.convolutions = [
            ConvolutionSequence(out_channels, convolutions_per_sequence)
            for _ in range(num_sequences)
        ]
        self.leaky_relu = LeakyReLU(leaky_relu_alpha)
        self.batch_normalizations = [BatchNormalization() for _ in range(num_sequences)]
        self.mesh_unpool = MeshUnpool() if unpool else None

    def call(
        self, mesh: Mesh, features: tf.Tensor, snapshot: CollapseSnapshot
    ) -> tf.Tensor:
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

        # Optionally run unpooling.
        if self.mesh_unpool is not None:
            return self.mesh_unpool(mesh, out_features, snapshot)
        return out_features
