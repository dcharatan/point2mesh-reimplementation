import numpy as np
from ...mesh.Mesh import Mesh


class CollapseSnapshot:
    """This holds information about which edges in a mesh were collapsed. It's
    used to make the pooling operation reversible.
    """

    # This is the mesh this CollapseSnapshot references.
    mesh: Mesh

    # This square matrix with shape (num_edges, num_edges) holds information
    # about how the mesh's edges have been collapsed. A nonzero entry at (i, j)
    # indicates that the ith edge is the parent of the jth edge. Note that all
    # edges start parented to themselves.
    relationships: np.ndarray

    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh

        # The identity matrix parents all edges to themselves.
        self.relationships = np.eye(mesh.edges.shape[0])

    def reparent(self, child_edge_key: int, parent_edge_key: int) -> None:
        """Reparent the child edge and its children to the parent edge."""
        self.relationships[parent_edge_key, :] += self.relationships[child_edge_key, :]
        self.relationships[child_edge_key, :] = 0