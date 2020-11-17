import unittest
import trimesh
import numpy as np
from typing import Tuple
from ..Mesh import Mesh


class TestMesh(unittest.TestCase):
    def load_obj(self, file_name: str) -> Tuple[np.ndarray]:
        with open(file_name, "r") as f:
            mesh = trimesh.exchange.obj.load_obj(f)
        return mesh["vertices"], mesh["faces"]

    def test_icosahedron_smoke(self):
        vertices, faces = self.load_obj("data/objs/icosahedron.obj")
        Mesh(vertices, faces)

    def test_icosahedron_edge_count(self):
        vertices, faces = self.load_obj("data/objs/icosahedron.obj")
        mesh = Mesh(vertices, faces)
        self.assertEqual(mesh.edges.shape, (30, 2))

    def test_icosahedron_vertex_degree(self):
        vertices, faces = self.load_obj("data/objs/icosahedron.obj")
        mesh = Mesh(vertices, faces)

        vertex_degrees = {}
        for edge_index in range(mesh.edges.shape[0]):
            v0, v1 = mesh.edges[edge_index, :]
            vertex_degrees[v0] = vertex_degrees.get(v0, 0) + 1
            vertex_degrees[v1] = vertex_degrees.get(v1, 0) + 1

        for degree in vertex_degrees.values():
            self.assertEqual(degree, 5)

    def test_icosahedron_lookup(self):
        vertices, faces = self.load_obj("data/objs/icosahedron.obj")
        mesh = Mesh(vertices, faces)

        # If you don't understand how the edge_lookup array works, read through this test carefully.
        for edge_index in range(mesh.edges.shape[0]):
            # Get the neighboring edges' keys (indices).
            neighbors = mesh.edge_to_neighbors[edge_index, :]

            # Get this edge's lookup table (4 entries).
            lookup = mesh.edge_lookup[edge_index, :]

            # Check that each neighbor's nth neighbor is the original edge, where n = lookup[neighbor index].
            for neighbor_index, neighbor in enumerate(neighbors):
                self.assertEqual(
                    edge_index, mesh.edge_to_neighbors[neighbor, lookup[neighbor_index]]
                )
