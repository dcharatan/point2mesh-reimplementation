from source.mesh.MeshChecker import MeshChecker
import unittest
import trimesh
import tensorflow as tf
import numpy as np
from ....mesh.Mesh import Mesh
from ..MeshPool import MeshPool
from ..CollapseSnapshot import CollapseSnapshot


class TestMeshPool(unittest.TestCase):
    def setUp(self) -> None:
        with open("data/objs/icosahedron.obj", "r") as f:
            mesh = trimesh.exchange.obj.load_obj(f)
        self.mesh = Mesh(mesh["vertices"], mesh["faces"])

    def test_mesh_remains_valid(self):
        num_edges = self.mesh.edges.shape[0]
        self.mesh_pool = MeshPool(num_edges - 1)

        initial_feature_values = np.zeros((num_edges, 3))
        initial_feature_values[:, :] = np.arange(num_edges)[:, None]

        features = tf.Variable(initial_feature_values, dtype=tf.float32)
        self.mesh_pool(self.mesh, features)

        checker = MeshChecker(self.mesh)
        self.assertTrue(checker.check_validity())

    def test_weighted_average(self):
        num_edges = self.mesh.edges.shape[0]
        self.mesh_pool = MeshPool(num_edges // 2)

        features = tf.Variable(np.ones((num_edges, 7)), dtype=tf.float32)
        new_features, _ = self.mesh_pool(self.mesh, features)

        self.assertTrue(np.allclose(new_features.numpy(), 1))
