import unittest
import trimesh
import tensorflow as tf
import numpy as np
from ....mesh.Mesh import Mesh
from ..Encoder import Encoder


class TestEncoder(unittest.TestCase):
    def setUp(self) -> None:
        with open("data/objs/icosahedron.obj", "r") as f:
            mesh = trimesh.exchange.obj.load_obj(f)
        self.mesh = Mesh(mesh["vertices"], mesh["faces"])

    def test_output_shape(self):
        num_edges = self.mesh.edges.shape[0]
        encoder = Encoder((12, 24, 48), 3, 4, 0.1, (27, 24, None))
        initial_feature_values = np.random.random((num_edges, 6))
        in_features = tf.Variable(initial_feature_values, dtype=tf.float32)

        out_features, _ = encoder(self.mesh, in_features)

        self.assertEqual(out_features.shape, (24, 48))

    def test_num_snapshots(self):
        num_edges = self.mesh.edges.shape[0]
        encoder = Encoder((12, 24, 48), 3, 4, 0.1, (27, 24, None))
        initial_feature_values = np.random.random((num_edges, 6))
        in_features = tf.Variable(initial_feature_values, dtype=tf.float32)

        _, snapshots = encoder(self.mesh, in_features)

        self.assertEqual(2, len(snapshots))
