import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from ...mesh.Mesh import Mesh
from ..feature import features_valid


class MeshConvolution(Layer):
    """MeshConvolution implements MeshCNN's mesh convolution operation by
    computing convolution between edges and 4 incident (1-ring) edge neighbors.
    """

    def __init__(
        self,
        out_channels,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    ):
        Layer.__init__(self)
        self.conv = tf.keras.layers.Conv2D(
            filters=out_channels,
            kernel_size=(1, 5),
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )

    def call(self, mesh: Mesh, features: tf.Tensor) -> tf.Tensor:
        """Compute convolution between edges and 4 incident (1-ring) edge
        neighbors using standard Conv2D.
        """
        assert features_valid(mesh, features)
        feature_image = self.create_feature_image(mesh, features)
        edge_feats = self.conv(feature_image)
        return edge_feats[0, :, 0, :]

    def create_feature_image(self, mesh: Mesh, features: tf.Tensor) -> tf.Tensor:
        """Using the connectivity information in mesh.edge_to_neighbor, create a
        an "image" that can be used by conv2d. The image's dimensions shape is
        (1, num_edges, 5, feature_size). Think of feature_size as in_channels.
        """

        # Create neighborhoods of the original edge plus its neighbors.
        neighborhoods = np.concatenate(
            [np.arange(mesh.edges.shape[0])[:, None], mesh.edge_to_neighbors], axis=1
        )

        # Gather features into the shape (num_edges, 5, feature_size).
        f = tf.gather(features, neighborhoods, axis=0)

        # Apply the symmetric functions to make an convolution equivariant.
        x_1 = f[:, 1, :] + f[:, 3, :]
        x_2 = f[:, 2, :] + f[:, 4, :]
        x_3 = tf.math.abs(f[:, 1, :] - f[:, 3, :])
        x_4 = tf.math.abs(f[:, 2, :] - f[:, 4, :])
        image = tf.stack([f[:, 0, :], x_1, x_2, x_3, x_4], axis=1)

        # Add a fake batch dimension.
        return image[None, :, :, :]
