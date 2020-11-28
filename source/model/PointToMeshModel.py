from tensorflow.keras import Model
from ..layers.composite.Encoder import Encoder
from ..layers.composite.Decoder import Decoder
from ..layers.convolution.MeshConvolution import MeshConvolution
from ..mesh.Mesh import Mesh
import tensorflow as tf


class PointToMeshModel(Model):
    def __init__(self) -> None:
        super(PointToMeshModel, self).__init__()
        self.encoder = Encoder((16, 32, 64), 1, 1, 0.1, (None, None, None))
        self.decoder = Decoder((32, 16, 6), 1, 1, 0.1, (None, None, None))
        initializer = tf.keras.initializers.RandomUniform(-1e-8, 1e-8)
        self.final_convolution = MeshConvolution(6, initializer, initializer)

    def call(self, mesh: Mesh, fixed_input_features: tf.Tensor):
        assert tf.rank(fixed_input_features) == 2 and fixed_input_features.shape[1] == 6

        encoding, snapshots = self.encoder(mesh, fixed_input_features)
        decoding = self.decoder(mesh, encoding, snapshots)
        output = self.final_convolution(mesh, decoding)

        return output