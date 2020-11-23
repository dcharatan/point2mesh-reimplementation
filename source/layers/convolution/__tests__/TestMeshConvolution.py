from source.mesh.MeshChecker import MeshChecker
import unittest
import trimesh
import tensorflow as tf
import numpy as np
from ....mesh.Mesh import Mesh
from ..MeshConvolution import MeshConvolution


class TestMeshConvolution(unittest.TestCase):
    def setUp(self) -> None:
        with open("data/objs/icosahedron.obj", "r") as f:
            mesh = trimesh.exchange.obj.load_obj(f)
        self.mesh = Mesh(mesh["vertices"], mesh["faces"])

    def test_output_shape(self):
        num_edges = self.mesh.edges.shape[0]
        self.mesh_convolution = MeshConvolution(2)

        initial_feature_values = np.zeros((num_edges, 3))
        initial_feature_values[:, :] = np.arange(num_edges)[:, None]

        features = tf.Variable(initial_feature_values, dtype=tf.float32)
        result = self.mesh_convolution(self.mesh, features)

        self.assertEqual(result.shape, (30, 2))
