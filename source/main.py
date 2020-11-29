import numpy as np
import trimesh
import tensorflow as tf
from tensorflow.keras import Model
from mesh.Mesh import Mesh
from loss.loss import BeamGapLossLayer, ChamferLossLayer, discrete_project
from model.PointToMeshModel import PointToMeshModel
from model.get_vertex_features import get_vertex_features

# initialize device: use GPU
device = tf.device("/gpu:0")
# print('device: {}'.format(device))

# open a mesh
with open("data/objs/icosahedron.obj", "r") as f:
    mesh = trimesh.exchange.obj.load_obj(f)
    mesh = Mesh(mesh["vertices"], mesh["faces"])

# create the model.
model = PointToMeshModel()

# initialize some parameters
beamgap_iterations = 800
beamgap_modulo = 2


# input point cloud
with open("./data/point_clouds/elephant.pwn", "r") as f:
    points = np.loadtxt(f)
    convex_hull = trimesh.convex.convex_hull(points[:, :3])
input_xyz = tf.convert_to_tensor(points[:, :3], dtype=tf.float32)
input_normals = tf.convert_to_tensor(points[:, 3:], dtype=tf.float32)
# normalize point cloud
scale = max([max(input_xyz[:, i]) - min(input_xyz[:, i]) for i in range(3)])
input_xyz = input_xyz / scale
target_mins = [(max(input_xyz[:, i]) - min(input_xyz[:, i])) / -2.0 for i in range(3)]
translations = [(target_mins[i] - min(input_xyz[:, i])) for i in range(3)]
input_xyz = input_xyz + translations

# initial and final sample size
r0 = 15000
rk = len(input_xyz)

# fixed random features C_l
fixed_input_features = tf.Variable(tf.random.normal([len(mesh.edges), 6], stddev=0.1))
fixed_input_features = tf.convert_to_tensor(fixed_input_features)

# net, optimizer, rand_verts, scheduler = init_net(
#     mesh, device, opts
# )  # a function in network
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# chamfer distance
chamfer_distance = ChamferLossLayer()
# calculate beam gap loss
beamgap_loss = BeamGapLossLayer(device, discrete_project)
if beamgap_iterations > 0:
    print("beamgap on")
    in_features = tf.concat([input_xyz, input_normals], -1)
    beamgap_loss.update_points_masks(mesh, in_features)

# iterate over 1000 times
for i in range(1000):

    # for part_i, est_verts in enumerate(mesh.vertices):
    #     print(est_verts)
    # perfrom gradient descent on every subset
    with tf.GradientTape() as tape:
        # input the initial mesh and a fixed random input feature into the self-prior network
        # and output a differential displacement vector per edge
        features = model(mesh, fixed_input_features)
        # calculate differential displacement per vertex
        features_per_vertics = get_vertex_features(mesh, features)
        # update vertices to reconstruct the mesh
        mesh.vertices += features_per_vertics
        # interatively increase sample size
        num_samples = min(rk, r0 + i * tf.math.floor((rk - r0) / 1000))
        # sample from mesh surface and then use these samples to calculate loss
        recon_xyz, recon_normals = mesh.sample_surface(mesh.vertices, num_samples)
        # calc chamfer loss w/ normals
        xyz_chamfer_loss, normals_chamfer_loss = chamfer_distance(recon_xyz, input_xyz)
        # drive the mesh into deep cavities: prevent Chamfer distance from being trapped in local min
        if (i < beamgap_iterations) and (
            i % beamgap_modulo == 0
        ):  # not sure about the values
            loss = beamgap_loss(mesh)
        else:
            loss = xyz_chamfer_loss + (0.1 * normals_chamfer_loss)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if i % 5 == 0:
        with open(f"tmp_out_{str(i).zfill(3)}.obj", "w") as f:
            tmesh = trimesh.Trimesh(faces=mesh["face"], vertices=mesh["vertices"])
            f.write(trimesh.exchange.obj.export_obj(tmesh))