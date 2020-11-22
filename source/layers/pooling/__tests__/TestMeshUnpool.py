from source.mesh.MeshChecker import MeshChecker
import unittest
import trimesh
import tensorflow as tf
import numpy as np
from ....mesh.Mesh import Mesh
from ..MeshPool import MeshPool
from ..MeshUnpool import MeshUnpool
from ..CollapseSnapshot import CollapseSnapshot


class TestMeshUnpool(unittest.TestCase):
    def setUp(self) -> None:
        with open("data/objs/icosahedron.obj", "r") as f:
            mesh = trimesh.exchange.obj.load_obj(f)
        self.mesh = Mesh(mesh["vertices"], mesh["faces"])

    def test_mesh_remains_valid(self):
        num_edges = self.mesh.edges.shape[0]
        self.mesh_pool = MeshPool(num_edges // 2)
        self.mesh_unpool = MeshUnpool()

        original_features = tf.Variable(np.ones((7, num_edges)), dtype=tf.float32)
        pooled_features, snapshot = self.mesh_pool(self.mesh, original_features)
        self.mesh_unpool(self.mesh, pooled_features, snapshot)

        checker = MeshChecker(self.mesh)
        self.assertTrue(checker.check_validity())

    def test_weighted_average(self):
        num_edges = self.mesh.edges.shape[0]
        self.mesh_pool = MeshPool(num_edges // 2)
        self.mesh_unpool = MeshUnpool()

        original_features = tf.Variable(np.ones((7, num_edges)), dtype=tf.float32)
        pooled_features, snapshot = self.mesh_pool(self.mesh, original_features)
        unpooled_features = self.mesh_unpool(self.mesh, pooled_features, snapshot)

        self.assertTrue(
            np.allclose(original_features.numpy(), unpooled_features.numpy())
        )
