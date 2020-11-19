from source.mesh.EdgeConnection import EdgeConnection
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from ..mesh.Mesh import Mesh
from enum import Enum


class MeshPool(Layer):
    def __init__(self) -> None:
        pass

    def call(self, mesh: Mesh, features: tf.Tensor):
        pass