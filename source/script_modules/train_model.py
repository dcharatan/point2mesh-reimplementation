from source.model.get_vertex_features import get_vertex_features
import tensorflow as tf
import numpy as np
import trimesh
from ..mesh.Mesh import Mesh
from ..model.PointToMeshModel import PointToMeshModel

print("Training PointToMeshModel.")

# Load a mesh.
with open("data/point_clouds/elephant.pwn", "r") as f:
    points = np.loadtxt(f)
    convex_hull = trimesh.convex.convex_hull(points[:, :3])

with open("tmp_out.obj", "w") as f:
    f.write(trimesh.exchange.obj.export_obj(convex_hull))

# Create the model.
model = PointToMeshModel()

# Create the mesh.
mesh = Mesh(convex_hull.vertices, convex_hull.faces)

# Create some random inputs.
initial_feature_values = np.random.random((mesh.edges.shape[0], 6))
in_features = tf.convert_to_tensor(initial_feature_values, dtype=tf.float32)

original_vertices = mesh.vertices.copy()

# Test out differentiation with respect to meaningless loss.
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
for i in range(1000):
    with tf.GradientTape() as tape:
        features = model(mesh, initial_feature_values)

        vertex_offsets = get_vertex_features(mesh, features)
        new_vertices = original_vertices + vertex_offsets

        surface_sample = mesh.sample_surface(new_vertices, 10000)

        loss = tf.math.reduce_sum(surface_sample)
        print(loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

print("Done.")
