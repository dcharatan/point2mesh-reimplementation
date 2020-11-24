from tensorflow.keras import Model
from ..layers.composite.Encoder import Encoder
from ..layers.composite.Decoder import Decoder
from ..mesh.Mesh import Mesh
import tensorflow as tf


class PointToMeshModel(Model):
    def __init__(self) -> None:
        super(PointToMeshModel, self).__init__()
        self.encoder = Encoder((12, 24, 48), 3, 4, 0.1, (27, 24, None))
        self.decoder = Decoder((24, 12, 6), 3, 4, 0.1, (27, 24, None))

    def call(self, mesh: Mesh, fixed_input_features: tf.Tensor):
        assert tf.rank(fixed_input_features) == 2 and fixed_input_features.shape[1] == 6

        encoding, snapshots = self.encoder(mesh, fixed_input_features)
        output = self.decoder(mesh, encoding, snapshots)

        return output