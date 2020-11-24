import tensorflow as tf
import numpy as np
import trimesh
from ..mesh.Mesh import Mesh
from ..model.PointToMeshModel import PointToMeshModel

print("Training PointToMeshModel.")

# Load a mesh.
with open("data/objs/icosahedron.obj", "r") as f:
    mesh = trimesh.exchange.obj.load_obj(f)
    mesh = Mesh(mesh["vertices"], mesh["faces"])

# Create the model.
model = PointToMeshModel()

# Create some random inputs.
initial_feature_values = np.random.random((30, 6))
in_features = tf.convert_to_tensor(initial_feature_values, dtype=tf.float32)

# Test out differentiation with respect to meaningless loss.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
for i in range(1000):
    with tf.GradientTape() as tape:
        features = model(mesh, initial_feature_values)
        loss = tf.math.reduce_sum(tf.math.abs(features))
        print(loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

print("Done.")
