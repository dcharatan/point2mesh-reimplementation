import tensorflow as tf
from typing import List
from tensorflow.keras.layers import Layer
from ...mesh.Mesh import Mesh
from ..convolution.MeshConvolution import MeshConvolution
from ..feature import features_valid


class ConvolutionSequence(Layer):
    """A ConvolutionSequence is simply a number of sequential convolutions."""

    convolutions: List[MeshConvolution]

    def __init__(self, out_channels: int, num_convolutions: int):
        Layer.__init__(self)

        # Create num_convolutions MeshConvolution layers.
        self.convolutions = [
            MeshConvolution(out_channels) for _ in range(num_convolutions)
        ]

    def call(self, mesh: Mesh, features: tf.Tensor) -> tf.Tensor:
        assert features_valid(mesh, features)

        for convolution in self.convolutions:
            features = convolution(mesh, features)

        return features
