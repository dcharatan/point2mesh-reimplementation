import unittest
from tensorflow.python.keras.regularizers import get
import trimesh
import numpy as np
from typing import Tuple
import tensorflow as tf
from ..ChamferLossLayer import ChamferLossLayer


class TestChamferLoss(unittest.TestCase):
    def load_obj_into_mesh(self, file_name: str) -> Tuple[np.ndarray]:
        with open(file_name, "r") as f:
            mesh = trimesh.exchange.obj.load_obj(f)
        return np.float32(mesh["vertices"]), mesh["faces"]

    def test_chamfer_loss_test(self):
        chamfer_loss_layer = ChamferLossLayer()
        cloud1 = tf.convert_to_tensor([[1.0, 1.0, 1.0]], dtype=tf.float32)
        cloud2 = tf.convert_to_tensor(
            [[0.0, 0.0, 0.0], [2.0, 1.0, 1.0], [2.0, 3.0, 4.0]],
            dtype=tf.float32,
        )
        # This is a bidirectional average square loss.
        # It should be (1)^2 + (3+1^2+(1+2^2+3^2))/3 = 1 + 6 = 7
        loss = chamfer_loss_layer(cloud1, cloud2)
        self.assertTrue(loss.numpy() - 7.0 < 0.01)
