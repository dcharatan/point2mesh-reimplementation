from ..Mesh import Mesh


class MeshChecker:
    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh

    def check_validity(self) -> bool:
        return (
            self.check_lookup_validity()
            and self.check_masked_edges_are_not_neighbors()
            and self.check_masked_edges_are_not_in_vertex_to_edges()
            and self.check_masked_vertices_are_not_in_vertex_to_edges()
            and self.check_vertex_to_edges_validity()
        )

    def check_vertex_to_edges_validity(self) -> bool:
        for vertex_index, edge_connections in enumerate(self.mesh.vertex_to_edges):
            if edge_connections is None:
                if self.mesh.vertex_mask[vertex_index]:
                    return False
                continue

            for edge_connection in edge_connections:
                expected_vertex_index = self.mesh.edges[
                    edge_connection.edge_index, edge_connection.index_in_edge
                ]
                if not expected_vertex_index == vertex_index:
                    return False
                if not self.mesh.edge_mask[edge_connection.edge_index]:
                    return False
        return True

    def check_masked_vertices_are_not_in_vertex_to_edges(self) -> bool:
        for vertex in range(self.mesh.vertices.shape[0]):
            is_masked = not self.mesh.vertex_mask[vertex]
            if is_masked and vertex in self.mesh.vertex_to_edges:
                return False
        return True

    def check_masked_edges_are_not_in_vertex_to_edges(self) -> bool:
        for index, edge_connections in enumerate(self.mesh.vertex_to_edges):
            if edge_connections is None:
                if self.mesh.vertex_mask[index]:
                    return False
                continue

            for edge_connection in edge_connections:
                if not self.mesh.edge_mask[edge_connection.edge_index]:
                    return False
        return True

    def check_masked_edges_are_not_neighbors(self) -> bool:
        """Return false if any unmasked edge's neighbor is masked."""
        for edge_key in range(self.mesh.edges.shape[0]):
            # Skip masked edges.
            if not self.mesh.edge_mask[edge_key]:
                continue

            # Make sure no neighbors are masked.
            neighbors = self.mesh.edge_to_neighbors[edge_key, :]
            for neighbor_key in neighbors:
                if not self.mesh.edge_mask[neighbor_key]:
                    return False
        return True

    def check_lookup_validity(self) -> bool:
        """Return false if edge_to_neighbors and edge_lookup disagree for an
        unmasked edge.
        """
        for edge_key in range(self.mesh.edges.shape[0]):
            # Skip masked edges.
            if not self.mesh.edge_mask[edge_key]:
                continue

            # Get the neighboring edges' keys (indices).
            neighbors = self.mesh.edge_to_neighbors[edge_key, :]

            # Get this edge's lookup table (4 entries).
            lookup = self.mesh.edge_lookup[edge_key, :]

            # Check that each neighbor's nth neighbor is the original edge, where n = lookup[neighbor index].
            for neighbor_index, neighbor in enumerate(neighbors):
                expected_edge_key = self.mesh.edge_to_neighbors[
                    neighbor, lookup[neighbor_index]
                ]
                if edge_key != expected_edge_key:
                    return False
        return True
