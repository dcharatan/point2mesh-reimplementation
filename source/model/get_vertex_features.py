from ..mesh.Mesh import Mesh
from ..layers.feature import features_valid
import tensorflow as tf
import numpy as np


def get_vertex_features(mesh: Mesh, features: tf.Tensor):
    """Convert edge features in shape (num_edges, 6) to vertex features in shape
    (num_vertices, 3).
    """
    assert features_valid(mesh, features)
    assert features.shape[1] == 6

    # Reshape features into the following shape:
    # (num_edges, num_vertices_per_edge = 2, dimensions_per_vertex = 3)
    edge_features = tf.reshape(features, (-1, 2, 3))

    # Now, pad the first two dimensions with slices of zeros.
    # This will be needed to handle variable vertex degrees using gather_nd.
    edge_features = tf.pad(edge_features, ((0, 1), (0, 1), (0, 0)))

    # Since TensorFlow doesn't have NumPy like indexing, it's a bit harder to do
    # this than in PyTorch.
    gathered_vertex_features = tf.gather_nd(edge_features, mesh.vertex_to_edges_tensor)
    gathered_vertex_features = tf.reshape(
        gathered_vertex_features, (mesh.vertices.shape[0], mesh.max_vertex_degree, 3)
    )

    # Get a weighted average of the features accumulated from the edges.
    vertex_features = tf.math.reduce_sum(gathered_vertex_features, axis=1)
    vertex_features /= mesh.vertex_to_degree[:, None]

    return vertex_features


def naive_get_vertex_features(mesh: Mesh, features: tf.Tensor):
    """This is a naive version of get_vertex_features used for testing."""
    assert features_valid(mesh, features)
    assert features.shape[1] == 6

    # Create a list of features for each vertex.
    vertex_features = [[] for _ in range(mesh.vertices.shape[0])]
    for edge_key in range(features.shape[0]):
        left_vertex, right_vertex = mesh.edges[edge_key, :]
        vertex_features[left_vertex].append(features[edge_key, :3])
        vertex_features[right_vertex].append(features[edge_key, 3:])

    # Use the mean of each vertex's list of features as that vertex's feature.
    vertex_features = [tf.stack(vf) for vf in vertex_features]
    vertex_features = [tf.math.reduce_mean(vf, axis=0) for vf in vertex_features]
    return tf.stack(vertex_features)