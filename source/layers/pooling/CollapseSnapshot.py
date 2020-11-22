from typing import Optional, List, Set
import numpy as np
from ...mesh.Mesh import Mesh
from ...mesh.EdgeConnection import EdgeConnection


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

    # These are copies of the mesh's fields.
    vertices: np.ndarray  # (num_vertices, 3)
    edges: np.ndarray  # (num_edges, 2)
    edge_to_neighbors: np.ndarray  # (num_edges, 4)
    edge_lookup: np.ndarray  # (num_edges, 4)
    vertex_to_edges: List[Optional[Set[EdgeConnection]]]

    def __init__(self, mesh: Mesh) -> None:
        # Assume we're starting with a clean mesh, i.e. no pooling/subdivision
        # has occurred.
        assert (mesh.edge_mask == True).all()
        assert (mesh.vertex_mask == True).all()

        # Save the information needed to reconstruct the mesh.
        # Note that this does not include the masks (see above) or the faces.
        # The faces are not included because collapse_edge does not modify them.
        self.vertices = np.copy(mesh.vertices)
        self.edges = np.copy(mesh.edges)
        self.edge_to_neighbors = np.copy(mesh.edge_to_neighbors)
        self.edge_lookup = np.copy(mesh.edge_lookup)
        self.vertex_to_edges = [set(ec) for ec in mesh.vertex_to_edges]

        # The identity matrix parents all edges to themselves.
        self.relationships = np.eye(mesh.edges.shape[0])

    def reparent(self, child_edge_key: int, parent_edge_key: int) -> None:
        """Reparent the child edge and its children to the parent edge."""
        self.relationships[parent_edge_key, :] += self.relationships[child_edge_key, :]
        self.relationships[child_edge_key, :] = 0