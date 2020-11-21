import unittest
import trimesh
import numpy as np
from typing import Tuple
from ....mesh.MeshChecker import MeshChecker
from ....mesh.Mesh import Mesh
from ..collapse_edge import check_collapse_manifold, collapse_edge


class TestCollapseEdge(unittest.TestCase):
    def load_obj(_, file_name: str) -> Tuple[np.ndarray]:
        with open(file_name, "r") as f:
            mesh = trimesh.exchange.obj.load_obj(f)
        return mesh["vertices"], mesh["faces"]

    def test_check_collapse_manifold_for_invalid(self):
        vertices, faces = self.load_obj("data/objs/tetrahedron.obj")
        mesh = Mesh(vertices, faces)
        for edge_key in range(mesh.edges.shape[0]):
            self.assertFalse(check_collapse_manifold(mesh, edge_key))

    def test_check_collapse_manifold_for_valid(self):
        vertices, faces = self.load_obj("data/objs/icosahedron.obj")
        mesh = Mesh(vertices, faces)
        for edge_key in range(mesh.edges.shape[0]):
            self.assertTrue(check_collapse_manifold(mesh, edge_key))

    def test_collapse_individual_icosahedron_edges(self):
        vertices, faces = self.load_obj("data/objs/icosahedron.obj")
        num_edges = Mesh(vertices, faces).edges.shape[0]
        for edge_key in range(num_edges):
            mesh = Mesh(vertices, faces)
            checker = MeshChecker(mesh)
            self.assertTrue(collapse_edge(mesh, edge_key))
            self.assertTrue(checker.check_validity())

    def test_collapse_sequential_icosahedron_edges(self):
        vertices, faces = self.load_obj("data/objs/icosahedron.obj")
        mesh = Mesh(vertices, faces)
        checker = MeshChecker(mesh)
        edge_key = 0
        for _ in range(mesh.edges.shape[0]):
            collapse_edge(mesh, edge_key)
            self.assertTrue(checker.check_validity())
            edge_key += 1
