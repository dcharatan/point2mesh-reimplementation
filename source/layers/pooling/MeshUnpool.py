import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from ...mesh.Mesh import Mesh
from ..feature import features_valid
from .collapse_edge import collapse_edge
from .CollapseSnapshot import CollapseSnapshot


class MeshUnpool(Layer):
    """MeshUnpool implements MeshCNN's mesh unpooling. It reverses the edge
    collapses carried out by MeshPool and distributes each edge's features to
    its children in the snapshot's relationships.
    """

    def __init__(self) -> None:
        Layer.__init__(self)

    def call(self, mesh: Mesh, features: tf.Tensor, snapshot: CollapseSnapshot):
        assert features_valid(mesh, features)

        # Restore the mesh's pre-collapse state.
        mesh.vertices = snapshot.vertices
        mesh.edges = snapshot.edges
        mesh.edge_to_neighbors = snapshot.edge_to_neighbors
        mesh.edge_lookup = snapshot.edge_lookup
        mesh.vertex_to_edges = snapshot.vertex_to_edges
        mesh.edge_mask = np.ones((mesh.edges.shape[0]), dtype=np.bool)
        mesh.vertex_mask = np.ones((mesh.vertices.shape[0]), dtype=np.bool)
        mesh.num_edges = mesh.edges.shape[0]

        # Use the snapshot's relationships to unpool the features.
        return self.unpool_features(features, snapshot)

    def unpool_features(
        self, features: tf.Tensor, snapshot: CollapseSnapshot
    ) -> tf.Tensor:
        """Use the snapshot's relationships to unpool the features. Each new
        edge's feature is simply its parent's feature.
        """
        return snapshot.extract_relationships().T @ features
