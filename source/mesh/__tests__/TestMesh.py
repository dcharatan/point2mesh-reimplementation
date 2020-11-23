import unittest
import trimesh
import numpy as np
import tensorflow as tf
from typing import Tuple
from ..MeshChecker import MeshChecker
from ..Mesh import Mesh
from ...layers.pooling.CollapseSnapshot import CollapseSnapshot
from ...layers.pooling.collapse_edge import collapse_edge


class TestMesh(unittest.TestCase):
    def load_obj(self, file_name: str) -> Tuple[np.ndarray]:
        with open(file_name, "r") as f:
            mesh = trimesh.exchange.obj.load_obj(f)
        return np.float32(mesh["vertices"]), mesh["faces"]

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

    def test_icosahedron_validity(self):
        vertices, faces = self.load_obj("data/objs/icosahedron.obj")
        mesh = Mesh(vertices, faces)
        checker = MeshChecker(mesh)
        self.assertTrue(checker.check_validity())

    def test_tetrahedron_validity(self):
        vertices, faces = self.load_obj("data/objs/tetrahedron.obj")
        mesh = Mesh(vertices, faces)
        checker = MeshChecker(mesh)
        self.assertTrue(checker.check_validity())

    def test_collapse_masked_elements(self):
        vertices, faces = self.load_obj("data/objs/icosahedron.obj")
        mesh = Mesh(vertices, faces)
        snapshot = CollapseSnapshot(mesh)
        checker = MeshChecker(mesh)
        for edge_key in range(mesh.edges.shape[0]):
            collapse_edge(mesh, edge_key, snapshot)
        mesh.collapse_masked_elements()
        self.assertTrue(checker.check_validity())

    def test_face_areas_normals(self):
        vertices, faces = self.load_obj("data/objs/icosahedron.obj")
        mesh = Mesh(vertices, faces)
        normals, areas = mesh.generate_face_areas_normals(mesh.vertices)
        checker = MeshChecker(mesh)
        self.assertTrue(np.abs(np.mean(areas.numpy()) - 4787.286) < 0.01)
        self.assertTrue(checker.check_validity)

    def test_sample(self):
        vertices, faces = self.load_obj("data/objs/icosahedron.obj")
        mesh = Mesh(vertices, faces)
        points, normals = mesh.sample_surface(tf.convert_to_tensor(mesh.vertices), 1001)
        self.assertTrue(np.abs(np.mean(points.numpy())) < 5)
        self.assertTrue(np.abs(np.mean(normals.numpy())) < 0.5)
