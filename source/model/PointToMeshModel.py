from tensorflow.keras import Model
from ..layers.composite.Encoder import Encoder
from ..layers.composite.Decoder import Decoder
from ..layers.convolution.MeshConvolution import MeshConvolution
from ..mesh.Mesh import Mesh
from typing import List
import tensorflow as tf


class PointToMeshModel(Model):
    def __init__(self, initial_num_faces: int, pooling: List[float]) -> None:
        super(PointToMeshModel, self).__init__()
        assert len(pooling) == 6
        scaled_pooling = [None if p is None else initial_num_faces * p for p in pooling]
        self.encoder = Encoder((6, 16, 32, 64, 64, 128), 1, 3, 0.01, scaled_pooling)
        self.decoder = Decoder((64, 64, 32, 16, 6, 6), 1, 1, 0.01, scaled_pooling)
        self.batch_normalization = tf.keras.layers.BatchNormalization()
        initializer = tf.keras.initializers.RandomUniform(-1e-8, 1e-8)
        self.final_convolution = MeshConvolution(6, initializer, initializer)

    def call(self, mesh: Mesh, fixed_input_features: tf.Tensor):
        assert tf.rank(fixed_input_features) == 2 and fixed_input_features.shape[1] == 6

        encoding, snapshots = self.encoder(mesh, fixed_input_features)
        decoding = self.decoder(mesh, encoding, snapshots)
        normalized = self.batch_normalization(decoding, training=True)
        output = self.final_convolution(mesh, normalized)

        return output