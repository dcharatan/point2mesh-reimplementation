import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from ...mesh.Mesh import Mesh
from ..feature import features_valid
from .collapse_edge import collapse_edge
from .CollapseSnapshot import CollapseSnapshot


class MeshPool(Layer):
    """MeshPool implements MeshCNN's mesh pooling. It collapses edges until a
    specified number of edges is reached. Each MeshPool should be paired with
    a corresponding MeshUnpool. Note that MeshPool's collapse operation does not
    update a mesh's faces (i.e. the mesh is valid except for its faces). This is
    because the edge structure, which is derived from the face structure, is all
    that is needed for pooling and un-pooling to work.
    """

    # Edge collapsing continues until the mesh's number of edges is edge_target.
    edge_target: int

    def __init__(self, edge_target: int) -> None:
        Layer.__init__(self)
        self.edge_target = edge_target

    def call(self, mesh: Mesh, features: tf.Tensor):
        assert features_valid(mesh, features)

        # Create a snapshot of the mesh's structure.
        # This contains all the information needed to undo the pooling (edge
        # collapse) operation. It also contains information about how the old
        # edges/features are mapped to the new ones in the relationships matrix.
        snapshot = CollapseSnapshot(mesh)

        # Sort the features by their L2 norms.
        values = tf.norm(features, axis=0)
        sorted_edge_keys = tf.argsort(values, direction="DESCENDING")

        # Collapse edges until edge_target is hit.
        # Keep track of how edges were merged in the snapshot's relationships
        # matrix, which will be used to determine how features will be pooled
        # and later unpooled.
        for edge_key in sorted_edge_keys:
            collapse_edge(mesh, edge_key.numpy(), snapshot)
            if mesh.num_edges <= self.edge_target:
                break

        # Rebuild the mesh to remove masked vertices and edges.
        mesh.collapse_masked_elements()

        # Return both the pooled features and the information needed to unpool.
        return self.pool_features(features, snapshot), snapshot

    def pool_features(
        self, features: tf.Tensor, snapshot: CollapseSnapshot
    ) -> tf.Tensor:
        """Use the snapshot's relationships to pool the features. Each new
        edge's feature is a weighted average of its children's features.
        """
        relationships = snapshot.extract_relationships()
        weighted = relationships / np.sum(relationships, axis=1)[:, None]
        return weighted @ features
