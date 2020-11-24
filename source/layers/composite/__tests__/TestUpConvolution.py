import unittest
import trimesh
import tensorflow as tf
import numpy as np
from ....mesh.Mesh import Mesh
from ..DownConvolution import DownConvolution
from ..UpConvolution import UpConvolution


class TestUpConvolution(unittest.TestCase):
    def setUp(self) -> None:
        with open("data/objs/icosahedron.obj", "r") as f:
            mesh = trimesh.exchange.obj.load_obj(f)
        self.mesh = Mesh(mesh["vertices"], mesh["faces"])

    def test_output_shape_with_pooling(self):
        num_edges = self.mesh.edges.shape[0]
        down = DownConvolution(17, 3, 4, 0.1, 18)
        up = UpConvolution(6, 3, 4, 0.1, True)
        initial_feature_values = np.random.random((num_edges, 6))
        in_features = tf.Variable(initial_feature_values, dtype=tf.float32)

        encoding, snapshot = down(self.mesh, in_features)
        out_features = up(self.mesh, encoding, snapshot)

        self.assertEqual(out_features.shape, (30, 6))

    def test_output_shape_without_pooling(self):
        num_edges = self.mesh.edges.shape[0]
        down = DownConvolution(17, 3, 4, 0.1, None)
        up = UpConvolution(6, 3, 4, 0.1, False)
        initial_feature_values = np.random.random((num_edges, 6))
        in_features = tf.Variable(initial_feature_values, dtype=tf.float32)

        encoding, snapshot = down(self.mesh, in_features)
        out_features = up(self.mesh, encoding, snapshot)

        self.assertEqual(out_features.shape, (30, 6))
