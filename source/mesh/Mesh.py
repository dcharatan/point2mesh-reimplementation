from random import sample
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops.gen_math_ops import arg_min
from .EdgeConnection import EdgeConnection
from typing import Optional, List, Set
import random


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
    edge_lookup: np.ndarray  # (num_edges, 4)

    # Map vertex indices to lists of edge indices.
    vertex_to_edges: List[Optional[Set[EdgeConnection]]]

    # This indicates whether an edge is still part of the mesh.
    edge_mask: np.ndarray

    # This indicates whether a vertex is still part of the mesh.
    vertex_mask: np.ndarray

    # The number of non-masked edges in the mesh.
    num_edges: int

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
        self.edge_mask = np.ones((self.edges.shape[0]), dtype=np.bool)
        self.vertex_mask = np.ones((self.vertices.shape[0]), dtype=np.bool)
        self.num_edges = self.edges.shape[0]

    def build_acceleration_structures(self):
        # Map vertex indices
        self.vertex_to_edges = [set() for _ in range(self.vertices.shape[0])]

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
                    self.vertex_to_edges[v0].add(EdgeConnection(edge_key, 0))
                    self.vertex_to_edges[v1].add(EdgeConnection(edge_key, 1))

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

    def collapse_masked_elements(self) -> None:
        """Rebuild the mesh without its masked elements. This creates a smaller
        mesh where nothing is masked off.
        """

        # Map each unmasked vertex to its new index.
        self.vertices = self.vertices[self.vertex_mask]
        new_vertex_indices = np.zeros_like(self.vertex_mask, dtype=np.int32)
        new_vertex_indices[self.vertex_mask] = np.arange(self.vertices.shape[0])

        # Update the edges. This requires two changes:
        # 1) Masked-off edges have to be removed.
        # 2) The vertex pair defining an edge has to be re-indexed to account
        #    for masked-off vertices that were removed.
        self.edges = new_vertex_indices[self.edges[self.edge_mask]]

        # Map each unmasked edge to its new index.
        new_edge_indices = np.zeros_like(self.edge_mask, dtype=np.int32)
        new_edge_indices[self.edge_mask] = np.arange(self.edges.shape[0])

        # Update edge_to_neighbors. This similarly requires two changes:
        # 1) Masked-off edges have to be removed.
        # 2) The neighbors have to be re-indexed to account for masked-off edges
        #    that were removed.
        self.edge_to_neighbors = new_edge_indices[
            self.edge_to_neighbors[self.edge_mask]
        ]

        # Update vertex_to_edges.
        new_vertex_to_edges = []
        for edge_connections in self.vertex_to_edges:
            # Remove masked-off vertices.
            if edge_connections is None:
                continue

            # Re-index edge keys (indices).
            new_edge_connections = {
                EdgeConnection(new_edge_indices[old.edge_index], old.index_in_edge)
                for old in edge_connections
            }
            new_vertex_to_edges.append(new_edge_connections)
        self.vertex_to_edges = new_vertex_to_edges

        # Update edge_lookup. This simply requires masked edges to be removed.
        self.edge_lookup = self.edge_lookup[self.edge_mask]

        # Reset the masks.
        self.edge_mask = np.ones((self.edges.shape[0],), dtype=np.bool)
        self.vertex_mask = np.ones((self.vertices.shape[0],), dtype=np.bool)

    def generate_face_areas_normals(self, vertices):
        """
        ARGS:
            vertices: a tensor of shape (num_verticies, 3) numpy array holding the XYZ coordinates of each vertex
            self.faces: a (num_faces, 3) numpy array containing the row index for all three verticies
                that make up a face
        return:
            face_unit_normals: a (num_faces, 3) numpy array represeting the XYZ components of the face's normal vector
            face_areas: a (num_faces) numpy array representing the area of the face
        """
        faces = self.faces

        # creates two (num_faces, 3) tensors where the rows are an XYZ vector representing one of the edge vectors
        edge_vectors_1 = tf.gather(vertices, faces[:, 1]) - tf.gather(
            vertices, faces[:, 0]
        )
        edge_vectors_2 = tf.gather(vertices, faces[:, 2]) - tf.gather(
            vertices, faces[:, 1]
        )

        # computes the cross product of the two edge arrays dim (num_faces, 3)
        edge_cross = tf.linalg.cross(edge_vectors_1, edge_vectors_2)

        # computes the magnitude of the adge_cross vector dim (numfaces)
        edge_cross_mag = tf.norm(edge_cross, axis=1)

        # the unit normal is the cross product divided by its magnitude dim (num_faces, 3)
        face_unit_normals = edge_cross / edge_cross_mag[:, None]

        # a triangle's area is equal to 1/2 * mag(edge cross product) dim (num_faces)
        face_areas = 0.5 * edge_cross_mag

        return face_unit_normals, face_areas

    def sample_surface(self, vertices, count):
        """
        ARGS:
            verticies: tensor of shape (num_verticies, 3) XYZ coordinate associated with each vertex
            self.faces: np array (num_faces, 3) set of three row indexesassociated with each face
            self.face_areas: np array (num_faces) hlding the area of each face
            self.face_unit_normals: np array (num_faces, 3) representing the XYZ vector of the unti normal for the face
            count: number of points to sample

        Uses:
        https://mathworld.wolfram.com/TrianglePointPicking.html

        returns:
            sample_points: tf (count, 3) XYZ coordinate for randomly sampled points accross the mesh
            sample_normals: tf (count, 3) XYZ norms for the selected points
        """

        faces = self.faces
        face_unit_normals, face_areas = self.generate_face_areas_normals(vertices)

        # normalize the face areas by the total area
        total_face_area = tf.math.reduce_sum(face_areas)
        face_areas = face_areas / total_face_area

        # Creates a probability distribution from the face areas
        face_distribution = tfp.distributions.Categorical(probs=face_areas)

        # samples the distribution count number of times, then gets the face row value for the relevant face
        face_to_sample = [face_distribution.sample() for i in range(count)]
        face_to_sample = tf.stack(face_to_sample)
        face_index = face_to_sample  # (count)
        face_to_sample = tf.gather(faces, face_to_sample)  # (count, 3)

        # sets XYZ "origins" for each triangle as the 0th index vertex (count, 3)
        origin = tf.gather(
            vertices,
            face_to_sample[:, 0],
        )

        # defines the two edges that, with the origin, define the triangle (count, 3)
        edge_1 = tf.gather(vertices, face_to_sample[:, 1]) - origin
        edge_2 = tf.gather(vertices, face_to_sample[:, 2]) - origin

        # stacks the two edge matricies together, dim (count, 2, 3)
        edges = tf.stack([edge_1, edge_2], axis=1)

        # computes two values between 0 and 1 that will scale the edge vectors and then summed to compute the points
        # dim (count, 2)
        edge_weights = tf.random.uniform(shape=(count, 2))

        # some points would be outside the triangle. if the sum is less than one then it is insde and thus
        # outside_triangle is true. dim (count)
        outside_triange = tf.math.reduce_sum(edge_weights, axis=1) > 1.0

        # remaps the points that are outside the trianle inside of the triangle
        edge_weights = edge_weights - tf.expand_dims(
            tf.cast(outside_triange, dtype=tf.float32), axis=1
        )
        edge_weights = tf.math.abs(edge_weights)

        # computes a sample vector as the weighted sum of the edge vectors using the random weights from above
        # dim (count, 3)
        sample_vector = tf.math.reduce_sum(
            edges * tf.expand_dims(edge_weights, axis=2), axis=1
        )

        # sample points are the displacement vector plus the origin
        # dim (count, 3)
        sample_points = sample_vector + origin

        # gather the normal for each point from the face it was sampled from
        sample_normals = tf.gather(face_unit_normals, face_index)

        return sample_points, sample_normals
