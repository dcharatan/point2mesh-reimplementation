import unittest
from tensorflow.python.keras.regularizers import get
import trimesh
import numpy as np
from typing import Tuple
import tensorflow as tf

from ..loss import (
    ChamferLossLayer,
    BeamGapLossLayer,
    discrete_project,
    get_looping_points,
    distance_within_cone,
)
from ...mesh.Mesh import Mesh


class TestLoss(unittest.TestCase):
    def load_obj_into_mesh(self, file_name: str) -> Tuple[np.ndarray]:
        with open(file_name, "r") as f:
            mesh = trimesh.exchange.obj.load_obj(f)
        return np.float32(mesh["vertices"]), mesh["faces"]

    def test_chamfer_loss_test(self):
        chamfer_loss_layer = ChamferLossLayer()

        # testing chamfer loss
        # defines two point clouds
        cloud1 = tf.convert_to_tensor([[1.0, 1.0, 1.0]], dtype=tf.float32)
        cloud2 = tf.convert_to_tensor(
            [[0.0, 0.0, 0.0], [2.0, 1.0, 1.0], [2.0, 3.0, 4.0]],
            dtype=tf.float32,
        )
        # bidirection, average square loss. Should be (1)^2 + (3+1^2+(1+2^2+3^2))/3 = 1 + 6 = 7
        loss = chamfer_loss_layer(cloud1, cloud2)
        self.assertTrue(loss.numpy() - 7.0 < 0.01)

    def test_initialize_beam_gap_loss_layer(self):
        init_test = BeamGapLossLayer("cpu", discrete_project)
        self.assertTrue(True)

    def test_update_point_masks(self):
        def testing_target_function(mesh, point_cloud, threshold):
            return (
                tf.convert_to_tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]]),
                tf.convert_to_tensor([True, False]),
            )

        test_layer = BeamGapLossLayer("cpu", testing_target_function)

        mesh = []
        target_point_cloud = []
        test_layer.update_points_masks(mesh, target_point_cloud)
        self.assertTrue(
            np.prod(
                np.equal(test_layer.points.numpy(), [[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]])
            )
        )
        self.assertTrue(np.prod(np.equal(test_layer.mask.numpy(), [True, False])))

    def test_looping_mask(self):
        cloud1 = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 10.0]]
        )
        cloud2 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 1.5]])
        cloud1_loop = get_looping_points(cloud1, cloud2)
        cloud2_loop = get_looping_points(cloud2, cloud1)
        self.assertTrue(np.prod(np.equal(cloud1_loop, [True, True, True, False])))
        self.assertTrue(np.prod(np.equal(cloud2_loop, [True, True, True])))

    def test_distance_within_cone(self):
        cloud1 = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 5.0], [1.0, 1.0, 1.0]])
        cloud1_normals = np.array(
            [[1.0, 0.0, 0.0], [0.0, -0.7071, -0.7071], [1.0, 0.0, 0.0]]
        )
        cloud2 = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        dist, mask = distance_within_cone(cloud1, cloud1_normals, cloud2, 0.99)
        test = np.sum(dist - np.array([[1.0, 2.0], [7.14, 7.35], [1.41, 1.73]]))
        self.assertTrue(
            np.sum(dist - np.array([[1.0, 2.0], [7.14, 7.35], [1.41, 1.73]])) < 0.1
        )
        self.assertTrue(
            np.prod(
                np.equal(mask, np.array([[True, True], [True, False], [False, False]]))
            )
        )

    def test_discrete_project(self):
        vertices, faces = self.load_obj_into_mesh("data/objs/icosahedron.obj")
        mesh = Mesh(vertices, faces)
        test_layer = BeamGapLossLayer("cpu", discrete_project)
        test_point_cloud, _ = mesh.sample_surface(
            tf.convert_to_tensor(mesh.vertices), 3
        )
        test_layer.update_points_masks(mesh, test_point_cloud)
        points = test_layer.points.numpy()
        mask = test_layer.mask.numpy()
        self.assertTrue(True)
