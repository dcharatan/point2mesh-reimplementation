from source.model.get_vertex_features import get_vertex_features
import tensorflow as tf
import numpy as np
import trimesh
from ..mesh.Mesh import Mesh
from ..model.PointToMeshModel import PointToMeshModel
from ..loss.ChamferLossLayer import ChamferLossLayer
from ..mesh.remesh import remesh

print("Training PointToMeshModel.")
physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load a mesh.
with open("data/point_clouds/elephant.pwn", "r") as f:
    points = np.loadtxt(f)
    convex_hull = trimesh.convex.convex_hull(points[:, :3])

# Create the model.
model = PointToMeshModel()

# Create the mesh.
remeshed_vertices, remeshed_faces = remesh(convex_hull.vertices, convex_hull.faces)
with open("tmp_out.obj", "w") as f:
    tmesh = trimesh.Trimesh(vertices=remeshed_vertices, faces=remeshed_faces)
    f.write(trimesh.exchange.obj.export_obj(tmesh))
mesh = Mesh(remeshed_vertices, remeshed_faces)

# Create some random inputs.
initial_feature_values = np.random.random((mesh.edges.shape[0], 6)) - 0.5
in_features = tf.convert_to_tensor(initial_feature_values, dtype=tf.float32)

loss_layer = ChamferLossLayer()

# Test out differentiation with respect to meaningless loss.
target_point_cloud = tf.convert_to_tensor(points[:, :3], dtype=tf.float32)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
current_vertices = remeshed_vertices
for i in range(1000):
    with tf.GradientTape() as tape:
        features = model(mesh, initial_feature_values)

        vertex_offsets = get_vertex_features(mesh, features)
        new_vertices = current_vertices + vertex_offsets

        surface_sample = mesh.sample_surface(new_vertices, 10000)

        chamfer_loss = loss_layer(surface_sample[0], target_point_cloud)

        print(chamfer_loss)

    if i % 5 == 0:
        with open(f"tmp_out_{str(i).zfill(3)}.obj", "w") as f:
            tmesh = trimesh.Trimesh(faces=remeshed_faces, vertices=new_vertices)
            f.write(trimesh.exchange.obj.export_obj(tmesh))

    gradients = tape.gradient(chamfer_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    mesh.vertices = current_vertices

print("Done.")
