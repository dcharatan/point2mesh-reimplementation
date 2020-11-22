import numpy as np
from .Mesh import Mesh


class MeshChecker:
    def __init__(self, mesh: Mesh) -> None:
        self.mesh = mesh

    def _check_vertices(self) -> bool:
        """Check the validity of the mesh's vertices field."""

        # Check the shape.
        if self.mesh.vertices.shape[1] != 3:
            return False

        # Create a list of vertices seen in faces.
        vertices_seen_in_faces = set()
        for vertex_a, vertex_b, vertex_c in self.mesh.faces:
            vertices_seen_in_faces.add(vertex_a)
            vertices_seen_in_faces.add(vertex_b)
            vertices_seen_in_faces.add(vertex_c)

        # Create a list of vertices seen in non-masked edges.
        vertices_seen_in_edges = set()
        for edge_key, edge in enumerate(self.mesh.edges):
            # Skip masked edges.
            if not self.mesh.edge_mask[edge_key]:
                continue
            vertices_seen_in_edges.add(edge[0])
            vertices_seen_in_edges.add(edge[1])

        # Check each vertex's validity.
        for vertex_key, xyz in enumerate(self.mesh.vertices):
            # Skip masked vertices.
            if not self.mesh.vertex_mask[vertex_key]:
                continue

            # The vertex's position should not include NaN.
            if np.isnan(xyz).any():
                return False

            # The vertex should appear in at least one edge.
            if vertex_key not in vertices_seen_in_edges:
                return False

            # The vertex should appear in at least one face.
            if vertex_key not in vertices_seen_in_faces:
                return False

            # The vertex should appear in vertex_to_edges.
            if self.mesh.vertex_to_edges[vertex_key] is None:
                return False

        return True

    def _check_edges(self) -> bool:
        """Check the validity of the mesh's edges field."""

        # Check the shape.
        if self.mesh.edges.shape[1] != 2:
            return False

        # Non-masked edges should not include masked vertices.
        for edge_key, edge in enumerate(self.mesh.edges):
            # Skip masked edges.
            if not self.mesh.edge_mask[edge_key]:
                continue
            a_masked = not self.mesh.vertex_mask[edge[0]]
            b_masked = not self.mesh.vertex_mask[edge[1]]
            if a_masked or b_masked:
                return False

        return True

    def _check_edge_to_neighbors_and_edge_lookup(self) -> bool:
        """Check the validity of the mesh's edge_to_neighbors and edge_lookup
        fields.
        """

        # Check the shapes.
        num_edges = self.mesh.edges.shape[0]
        if self.mesh.edge_to_neighbors.shape != (num_edges, 4):
            return False
        if self.mesh.edge_lookup.shape != (num_edges, 4):
            return False

        for edge_key, neighbors in enumerate(self.mesh.edge_to_neighbors):
            # Skip masked edges.
            if not self.mesh.edge_mask[edge_key]:
                continue

            # Confirm that edge_to_neighbors and edge_lookup agree.
            # Check that each neighbor's nth neighbor is the original edge,
            # where n = lookup[neighbor index].
            lookup = self.mesh.edge_lookup[edge_key, :]
            for neighbor_index, neighbor in enumerate(neighbors):
                expected_edge_key = self.mesh.edge_to_neighbors[
                    neighbor, lookup[neighbor_index]
                ]
                if edge_key != expected_edge_key:
                    return False

            # Make sure no neighbors are masked.
            neighbors = self.mesh.edge_to_neighbors[edge_key, :]
            for neighbor_key in neighbors:
                if not self.mesh.edge_mask[neighbor_key]:
                    return False

        return True

    def _check_vertex_to_edges(self):
        """Check the validity of the mesh's vertex_to_edges field."""

        for vertex_key, edge_connections in enumerate(self.mesh.vertex_to_edges):
            # A masked vertex should not have EdgeConnections.
            if edge_connections is None:
                if self.mesh.vertex_mask[vertex_key]:
                    return False
                continue

            # Check the validity of the EdgeConnections.
            for edge_connection in edge_connections:
                edge_key = edge_connection.edge_index
                index_in_edge = edge_connection.index_in_edge

                # EdgeConnections should not include masked edges.
                if not self.mesh.edge_mask[edge_key]:
                    return False

                # The EdgeConnection should agree with what's in edges.
                if self.mesh.edges[edge_key, index_in_edge] != vertex_key:
                    return False

        return True

    def _check_edge_mask(self) -> bool:
        """Check the validity of the mesh's edge_mask field."""

        num_edges = self.mesh.edges.shape[0]
        if self.mesh.edge_mask.shape != (num_edges,):
            return False

        return True

    def _check_vertex_mask(self) -> bool:
        """Check the validity of the mesh's vertex_mask field."""

        num_vertices = self.mesh.vertices.shape[0]
        if self.mesh.vertex_mask.shape != (num_vertices,):
            return False

        return True

    def check_validity(self) -> bool:
        return (
            True
            and self._check_vertices()
            and self._check_edges()
            and self._check_edge_to_neighbors_and_edge_lookup()
            and self._check_vertex_to_edges()
            and self._check_edge_mask()
            and self._check_vertex_mask()
        )
