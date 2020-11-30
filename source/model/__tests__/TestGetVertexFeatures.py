import unittest
import trimesh
import tensorflow as tf
import numpy as np
from ...mesh.Mesh import Mesh
from ..get_vertex_features import get_vertex_features, naive_get_vertex_features


class TestGetVertexFeatures(unittest.TestCase):
    def setUp(self) -> None:
        with open("data/objs/icosahedron.obj", "r") as f:
            mesh = trimesh.exchange.obj.load_obj(f)
        self.mesh = Mesh(mesh["vertices"], mesh["faces"])

    def test_against_naive(self):
        num_edges = self.mesh.edges.shape[0]
        features = tf.random.uniform((num_edges, 6))

        fast = get_vertex_features(self.mesh, features)
        naive = naive_get_vertex_features(self.mesh, features)

        tf.debugging.assert_near(fast, naive)
