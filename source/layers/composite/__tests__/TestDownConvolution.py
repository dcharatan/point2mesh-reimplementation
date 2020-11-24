import unittest
import trimesh
import tensorflow as tf
import numpy as np
from ....mesh.Mesh import Mesh
from ..DownConvolution import DownConvolution


class TestDownConvolution(unittest.TestCase):
    def setUp(self) -> None:
        with open("data/objs/icosahedron.obj", "r") as f:
            mesh = trimesh.exchange.obj.load_obj(f)
        self.mesh = Mesh(mesh["vertices"], mesh["faces"])

    def test_output_shape_with_pooling(self):
        num_edges = self.mesh.edges.shape[0]
        down = DownConvolution(17, 3, 4, 0.1, 18)
        initial_feature_values = np.random.random((num_edges, 6))
        in_features = tf.Variable(initial_feature_values, dtype=tf.float32)

        out_features, _ = down(self.mesh, in_features)

        self.assertEqual(out_features.shape, (18, 17))

    def test_output_shape_without_pooling(self):
        num_edges = self.mesh.edges.shape[0]
        down = DownConvolution(31, 3, 4, 0.1, None)
        initial_feature_values = np.random.random((num_edges, 2))
        in_features = tf.Variable(initial_feature_values, dtype=tf.float32)

        out_features, _ = down(self.mesh, in_features)

        self.assertEqual(out_features.shape, (30, 31))
