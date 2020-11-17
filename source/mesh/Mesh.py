import numpy as np
from .EdgeConnection import EdgeConnection
from typing import Dict, List


class Mesh:
    # Hold the mesh's vertices as xyz points.
    vertices: np.ndarray  # (num_vertices, 3)

    # Hold the mesh's faces as triplets of vertex indices.
    faces: np.ndarray  # (num_faces, 3)

    # Hold the mesh's edges as couples of vertex indices.
    # Note that an edge's key is its row index within this array.
    edges: np.ndarray  # (num_edges, 2)

    # Map each edge's key to those of its four neighbors.
    edge_to_neighbors: np.ndarray  # (num_edges, 4)

    # Maps each edge's key to that edge's index within other entries in edge_to_neighbors.
    # Let k be an edge's key. Let k'_n be the nth value of edge_to_neighbors[k].
    # Then the nth element of edge_lookup[k] will be the index of k within edge_to_neighbors[k'_n].
    # This is used for MeshCNN's pooling operation.
    edge_lookup: np.ndarray

    # Map vertex indices to lists of edge indices.
    vertex_to_edges: Dict[int, List[EdgeConnection]]

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        """Create a new mesh.

        vertices: the mesh's vertices with shape (num_vertices, 3).
        faces: the mesh's faces with shape (num_faces, 3).
        """
        assert isinstance(vertices, np.ndarray)
        assert vertices.shape[1] == 3
        assert isinstance(faces, np.ndarray)
        assert faces.shape[1] == 3
        self.vertices = vertices
        self.faces = faces
        self.build_acceleration_structures()

    def build_acceleration_structures(self):
        # Map vertex indices
        self.vertex_to_edges = [[] for _ in range(self.vertices.shape[0])]

        # Each edge has a unique key.
        # These map between (smaller vertex index, larger vertex index) and an edge's unique key.
        edge_to_key = {}
        key_to_edge = []

        # This maps each edge's key to its neighboring edges' keys.
        # In a watertight mesh, each edge has four neighbors.
        # The neighbors are the four other edges of the two triangles an edge is connected to.
        edge_to_neighbors: List[List[int]] = []

        # This maps each edge to the number of neighbors that have been encountered.
        edge_to_seen_neighbors: List[int] = []

        # This maps each edge's key to that edge's index within other entries in edge_to_neighbors.
        # Let k be an edge's key. Let k'_n be the nth value of edge_to_neighbors[k].
        # Then the nth element of edge_lookup[k] will be the index of k within edge_to_neighbors[k'_n].
        edge_lookup: List[List[int]] = []

        for face in self.faces:
            # Create a list of the face's edges.
            # Each entry is a sorted list of the edge's two vertex indices.
            face_edges = [tuple(sorted([face[i], face[(i + 1) % 3]])) for i in range(3)]

            # The stuff here happens once per edge (the first time it's encountered).
            for edge_vertices in face_edges:
                if edge_vertices not in edge_to_key:
                    # The edge gets a unique key (its index).
                    edge_key = len(edge_to_key)
                    edge_to_key[edge_vertices] = edge_key
                    key_to_edge.append(edge_vertices)

                    # Set up data structures for the new edge.
                    edge_to_neighbors.append([None, None, None, None])
                    edge_lookup.append([None, None, None, None])
                    edge_to_seen_neighbors.append(0)

                    # Associate each vertex with the edge.
                    v0, v1 = edge_vertices
                    self.vertex_to_edges[v0].append(EdgeConnection(edge_key, 0))
                    self.vertex_to_edges[v1].append(EdgeConnection(edge_key, 1))

            # Associate edges with their neighbors.
            # This happens in a separate loop because it requires all encountered edges to have keys.
            for edge_index, edge_vertices in enumerate(face_edges):
                # Get the edge's neighbors' keys.
                neighbor_0_key = edge_to_key[face_edges[(edge_index + 1) % 3]]
                neighbor_1_key = edge_to_key[face_edges[(edge_index + 2) % 3]]

                # Update edge_to_neighbors.
                edge_key = edge_to_key[edge_vertices]
                seen_neighbors = edge_to_seen_neighbors[edge_key]
                edge_to_neighbors[edge_key][seen_neighbors + 0] = neighbor_0_key
                edge_to_neighbors[edge_key][seen_neighbors + 1] = neighbor_1_key
                edge_to_seen_neighbors[edge_key] += 2

            # Create the edge lookup.
            # This happens in a separate loop because it requires all encountered edges' seen neighbor counts to be up to date.
            for edge_index, edge_vertices in enumerate(face_edges):
                # Get the edge's neighbors' keys.
                neighbor_0_key = edge_to_key[face_edges[(edge_index + 1) % 3]]
                neighbor_1_key = edge_to_key[face_edges[(edge_index + 2) % 3]]

                # Find how many neighbors the neighbors have seen.
                neighbor_0_seen_neighbors = edge_to_seen_neighbors[neighbor_0_key]
                neighbor_1_seen_neighbors = edge_to_seen_neighbors[neighbor_1_key]

                # Deduce the current edge's index in its neighbors' entries in edge_to_neighbors.
                # Note: The keys to edge_lookup here are the same as the keys to edge_to_neighbors in the previous loop.
                # They just look different because seen_neighbors was incremented by 2.
                edge_key = edge_to_key[edge_vertices]
                seen_neighbors = edge_to_seen_neighbors[edge_key]
                edge_lookup[edge_key][seen_neighbors - 2] = (
                    neighbor_0_seen_neighbors - 1
                )
                edge_lookup[edge_key][seen_neighbors - 1] = (
                    neighbor_1_seen_neighbors - 2
                )

        # Save the results to instance variables.
        self.edges = np.array(key_to_edge, dtype=np.int32)
        self.edge_to_neighbors = np.array(edge_to_neighbors, dtype=np.int64)
        self.edge_lookup = np.array(edge_lookup, dtype=np.int64)
