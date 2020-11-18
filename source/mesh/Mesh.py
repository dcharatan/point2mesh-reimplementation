import numpy as np
import tensorflow as tf
import scipy.spatial.KDTree as KDTree
from tensorflow.python.ops.gen_math_ops import arg_min
from .EdgeConnection import EdgeConnection
from typing import Optional, List, Set


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

def discrete_project(self, point_cloud, threshold=0.9):
    """Compute ....

    ARGS:
        self: a Mesh Object
        point_cloud: a tf tensor of shap (num_points_in_cloud, 3)
        threshold: a flot representing the cosine of the cone angle when checking to see if there exist any points
            within the target pointcloud that are relevant for the mash

    returns:
        pc_per_face: a (num_faces, 3) tensor that repersents the XYZ coordinates for the nearest point_cloud point to
            a given face's midpoints. Instead coordinates will be nan if:
                the face midpoint is looping (see below)
                there is no point cloud point within the cone (see below)
        pc_is_not_nan: a (num_faces) tensor that is True if the coordinates in a given row are not nan, alse otherwise
    """

    # Check that point_cloud is tensor with corrrect dimension size
    # Check that self is a Mesh
    assert tf.is_tensor(point_cloud)
    assert tf.shape(point_cloud)[-1] == 3
    assert isinstance(self, Mesh)

    # Cast to data type double
    # Also get numpy to stop gradient
    point_cloud = tf.cast(point_cloud, dtype=tf.double)  # (num_points_in_cloud, 3)
    point_cloud = point_cloud.numpy()

    # Get face normals for mesh, point normal to the mesh out of the face midpoints
    mesh_normals = self.normals  # (num_faces, 3)

    # Get set of 3 vertexes associated with each face.
    # Compute the midpoint of every face by taking the mean of each dimension over the 3 verticies
    # Also get numpy as to stop gradient
    mid_points = self.vertices[
        self.faces
    ]  # (num_faces, 3(3 vertexes per face), 3(3 coordinates per vertex))
    mid_points = tf.nn.reduce_mean(
        mid_points, axis=1
    ).numpy()  # (num_faces, 3(3 avg coordiates per face))

    # create a scipy KDTree to represent points
    # ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
    KDTree_cloud = KDTree(point_cloud)
    KDTree_mesh = KDTree(mid_points)

    # querries each KDTree to find the k nearest neighbors. As an example:
    # k_closest_cloud_points2mesh will have the shape (np.shape(mid_points)[0], k (in our case 3))
    # each row corresponds to a midpoint, the nth index in that row corresponds to the index row into point_cloud
    # that is nth closest to the given midpoint
    # The same is querried for the mesh cloud as well (can be thought of as bi directional)
    k_closest_cloud_points2mesh = KDTree_cloud.query(
        mid_points, k=3, eps=0
    )  # (num_faces, k)
    k_closest_mesh_points2cloud = KDTree_mesh.query(
        point_cloud, k=3, eps=0
    )  # (num_points_in_cloud, k)

    # gets, for every face mid_point in the mesh, the k nearest neighbors in the point cloud
    # then, for all k of those points in the point cloud, gets the k nearest points in the set of mid_points
    # finally, reshapes from (num_faces, 3, 3) to (num_faces, 9)

    # as an example if the meshes were identical (mid_points == point_cloud)
    # with points [0,1,2] in mid_points and [a,b,c] in point_cloud
    # where dist(0,1)<dist(0,2)<dist(1,2)
    # then neighbors_neighbors would apear as follows:
    # [
    # [0, 1, 2, 1, 0, 2, 2, 0, 1]
    # [1, 0, 2, 0, 1, 2, 2, 0, 1]
    # [2, 0, 1, 0, 1, 2, 1, 0, 2]
    # ]
    neighbors_neighbors = np.reshape(
        k_closest_cloud_points2mesh[k_closest_mesh_points2cloud],
        (np.shape(k_closest_cloud_points2mesh)[0], -1),
    )  # (num_faces, 9)

    # essentaially this checks to see if there are any adjacency loops
    # neighbors_neighbors == arange checks to see, for each mind_point n, are its neighbor's neighbors n?
    #   True if mapping back to itself, False if not.
    #   boolean array of shape (num_faces, 9)
    # sum along axis 1 to see how many times a point maps back to itself
    #   m number of maps back to itself, max(m) = 3
    #   int array of shape (num_faces)
    # check if remaps are > 0:
    #   True if a point maps back to itself at least once amoung 9 options
    #   boolean array of shape (num_faces)

    looping_mesh_points_mask = (
        np.sum(
            np.array(
                neighbors_neighbors
                == np.arange(0, np.shape(k_closest_cloud_points2mesh)[0])
            ),
            axis=1,
        )
        > 0
    )

    # We are only interested in points that DONT map back to themselves because this indicates that the mesh is not
    # entering a deep enough cavity. Thus we pull from mid_points and normals using ~ mask
    # From here on, we will call the number of non-looping points num_non_looping
    masked_mid_points = mid_points[~looping_mesh_points_mask, :]  # (num_non_looping, 3)
    masked_normals = mesh_normals[~looping_mesh_points_mask, :]  # (num_non_looping, 3)

    # This uses broadcasting to compute the displacement from every non-looping point in the mesh mid_points
    # to every point in the point cloud
    displacement = (
        masked_mid_points[:, None, :] - point_cloud
    )  # (num_non_looping, num_points_in_cloud, 3)

    # Compute distance via L2 norm on displacements
    distance = np.linalg.norm(
        displacement, axis=-1
    )  # (num_non_looping, num_points_in_cloud)

    # Another big, multistep operation.
    # First, normalize the displacement by its distance to compute the unit normal in the direction of the displacement
    # Second, compute the dot product of this norm with the mesh surface norm:
    #   If the dot product is close to 1 then these vectors are nearly parallel
    #   If the dot product is close to 0 then these vectors are perpendicular
    # Third, create the mask by checking if the dot product is close to 1 (within threshold)
    # In the end, boolean array of shape (num_non_looping)
    #   True if there is a point within the cone
    masked_within_cone = np.array(
        np.abs(
            np.sum(
                (displacement / distance[:, :, None]) * masked_normals[:, None, :],
                axis=-1,
            )
        )
        > threshold
    )

    # For every non looping point that doesn't have a corresponding point on the PC within cone
    # Set the distances between that mesh point and all the PC points = inf
    distance[~masked_within_cone] += float(
        "inf"
    )  # (num_non_looping, num_points_in_cloud)

    # compute the minumum for each point in num_non_looping of the distances to all the points in the point cloud
    # also compute the index of the point cloud point associated with the minimum distance
    min = np.amin(distance, axis=-1)  # (num_non_looping)
    arg_min = np.argmin(distance, axis=-1)  # (num_non_looping)

    # For each non looping point gets the XYZ coordniates of the closest point in the point cloud
    # Fot all points where the was no point in the point cloud within the cone, sets the XYZ coordinates to nan
    pc_per_masked_face = point_cloud[arg_min, :].copy()
    pc_per_masked_face[min == float("inf"), :] = float("nan")

    # creats one of the output objects as a tensor with shape (num_faces, 3)
    pc_per_face = tf.zeros(
        shape=(np.shape(mid_points)[0], 3), dtype=pc_per_masked_face.dtype
    )

    # for every non looping point, we have already computed the xyz point on the PC that is closest
    # so this just writes those values to be the percomputed values
    pc_per_face[~looping_mesh_points_mask, :] = pc_per_masked_face

    # for all the looping points, just set the closest poinct cloud XYZs to nan
    pc_per_face[looping_mesh_points_mask, :] = float("nan")

    # checks to see if the XYZ points are nan becasue nan=/=nan
    #   True indicates the XYZ point is real which happens when the corresponing mesh midpoint is non-looping AND
    #       there is a point on the point cloud within the cone
    pc_is_non_nan = pc_per_face[:, 0] == pc_per_face[:, 0]

    # return objects
    return pc_per_face, pc_is_non_nan
