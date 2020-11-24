import unittest
import trimesh
import tensorflow as tf
import numpy as np
from ....mesh.Mesh import Mesh
from ..ConvolutionSequence import ConvolutionSequence


class TestConvolutionSequence(unittest.TestCase):
    def setUp(self) -> None:
        with open("data/objs/icosahedron.obj", "r") as f:
            mesh = trimesh.exchange.obj.load_obj(f)
        self.mesh = Mesh(mesh["vertices"], mesh["faces"])

    def test_output_shape(self):
        num_edges = self.mesh.edges.shape[0]
        sequence = ConvolutionSequence(11, 3)

        initial_feature_values = np.random.random((num_edges, 3))
        features = tf.Variable(initial_feature_values, dtype=tf.float32)

        result = sequence(self.mesh, features)

        self.assertEqual(result.shape, (30, 11))
