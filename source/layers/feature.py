import tensorflow as tf
from ..mesh.Mesh import Mesh


def features_valid(mesh: Mesh, features: tf.Tensor) -> bool:
    """Check whether a feature tensor is compatible with the specified mesh."""

    # A feature tensor's shape should be (feature_length, num_edges).
    return tf.rank(features) == 2 and features.shape[0] == mesh.edges.shape[0]