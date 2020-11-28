from ..mesh.Mesh import Mesh
from ..layers.feature import features_valid
import tensorflow as tf


def get_vertex_features(mesh: Mesh, features: tf.Tensor):
    """Convert edge features in shape (num_edges, 6) to vertex features in shape
    (num_vertices, 3). If possible, this should be vectorized somehow.
    """
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
