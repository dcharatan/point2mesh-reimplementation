import numpy as np
import tensorflow as tf
import scipy
from tensorflow.keras.layers import Layer
from ..mesh.Mesh import Mesh


class BeamGapLossLayer(Layer):
    def __init__(self, device, target_function) -> None:

        super(BeamGapLossLayer, self).__init__()

        # Sets the device to use equal to device
        self.device = device

        # Sets the target function. In this case is mesh.discrete project except for debugging
        self.target_function = target_function

        # Initializes:
        #   self.points which is used to store the points that will eventually be used to calculate beam gap loss
        #   self.masks which is used to store the mask that will eventually be used to calculate beam gap loss
        self.points = None
        self.masks = None

    def update_points_masks(self, mesh, target_point_cloud):

        """
        Updates self.points and self.mask
        self.points: a (num_faces, 3) tensor that repersents the XYZ coordinates for the nearest point_cloud point to
            a given face's midpoints. Instead coordinates will be nan if:
                the face midpoint is looping (see below)
                there is no point cloud point within the cone (see below)
        self.mask: a (num_faces) tensor that is True if the coordinates in a given row are not nan, alse otherwise
        """
        self.points, self.mask = self.target_function(mesh, target_point_cloud, 0.99)

    def call(self, mesh):

        """
        ARGS:
            self: a beam gap loss layer
            mesh: a Mesh object
        return:
            l2: 10 times the loss which is computed by taking the distance from the desired points to the closest point in the point cloud

        """

        # losses is a (numfaces, 3) tensor where the XYZ coordinate is the the distance
        # from the closest point on the point cloud to the mid_point of a given face.
        # Note that the faces we want to ignore (looping or no PC point within cone) will have nan values here
        losses = self.points - tf.math.reduce_mean(mesh.vertices[mesh.faces], axis=1)

        # compute the "length" of the loss by computing norm of the XYZ
        # also only select the non-nan points for the loss
        losses = tf.norm(losses, axis=-1)[self.mask]  # (numfaces)

        # take the reduced mean of all the lengths of the losses and cast as float32. Mutliply by 10
        l2 = 10.0 * tf.cast(tf.math.reduce_mean(losses), dtype=tf.float32)  # ()

        return l2


def discrete_project(mesh, point_cloud, vertices=None, threshold=0.9):
    """
    ARGS:
        self: a Mesh Object
        vertices: a tf tensor of shape (num_vertices, 3)
        point_cloud: a tf tensor of shap (num_points_in_cloud, 3)
        threshold: a float representing the cosine of the cone angle when checking to see if there exist any points
            within the target pointcloud that are relevant for the mash

    returns:
        pc_per_face: a (num_faces, 3) tensor that repersents the XYZ coordinates for the nearest point_cloud point to
            a given face's midpoints. Instead coordinates will be nan if:
                the face midpoint is looping (see below)
                there is no point cloud point within the cone (see below)
        pc_is_not_nan: a (num_faces) tensor that is True if the coordinates in a given row are not nan, alse otherwise
    """

    # Check that point_cloud is tensor with corrrect dimension size
    # Check that self is a Mesh
    assert tf.is_tensor(point_cloud)
    if not tf.is_tensor(vertices):
        vertices = mesh.vertices
    else:
        vertices = vertices.numpy()

    assert tf.shape(point_cloud)[-1] == 3
    assert isinstance(mesh, Mesh)

    # Cast to data type double
    # Also get numpy to stop gradient
    point_cloud = tf.cast(point_cloud, dtype=tf.double)  # (num_points_in_cloud, 3)
    point_cloud = point_cloud.numpy()

    # Get face normals for mesh, point normal to the mesh out of the face midpoints
    mesh_normals, _ = mesh.generate_face_areas_normals(
        tf.convert_to_tensor(vertices)
    )  # (num_faces, 3)

    mesh_normals = mesh_normals.numpy()

    # Get set of 3 vertexes associated with each face.
    # Compute the midpoint of every face by taking the mean of each dimension over the 3 vertices
    # Also get numpy as to stop gradient
    mid_points = vertices[
        mesh.faces
    ]  # (num_faces, 3(3 vertexes per face), 3(3 coordinates per vertex))
    mid_points = tf.math.reduce_mean(
        mid_points, axis=1
    ).numpy()  # (num_faces, 3(3 avg coordiates per face))

    # See function below.
    # looping_mesh_points_mask: numpy boolean array of shape (num_faces) that is True if any of the k nearest points cloud
    #   points of the row index's mesh point has the original mesh point amoung its k nearest neighbors
    looping_mesh_points_mask = get_looping_points(mid_points, point_cloud)

    # We are only interested in points that DONT map back to themselves because this indicates that the mesh is not
    # entering a deep enough cavity. Thus we pull from mid_points and normals using ~ mask
    # From here on, we will call the number of non-looping points num_non_looping
    masked_mid_points = mid_points[~looping_mesh_points_mask, :]  # (num_non_looping, 3)
    masked_normals = mesh_normals[~looping_mesh_points_mask, :]  # (num_non_looping, 3)

    # See function below.
    # distance: numpy arrar of shape (num_non_looping, num_points) which is the l2 distance from each of the face points to each of the
    #   point cloud points
    # masked_within_cone: numpy boolean array of shape (num_non_looping, num_points) that is True if the (unit displacement between
    #   the face mid_point and the point cloud point) dotted with (the face unit normal) is absolute value greater than the
    #   threshold
    distance, masked_within_cone = distance_within_cone(
        masked_mid_points, masked_normals, point_cloud, threshold
    )

    # For every non looping point that doesn't have a corresponding point on the PC within cone
    # Set the distances between that mesh point and all the PC points = inf
    distance[~masked_within_cone] += float(
        "inf"
    )  # (num_non_looping, num_points_in_cloud)

    # compute the minumum for each point in num_non_looping of the distances to all the points in the point cloud
    # also compute the index of the point cloud point associated with the minimum distance
    min = np.amin(distance, axis=-1)  # (num_non_looping)
    arg_min = np.argmin(distance, axis=-1)  # (num_non_looping)

    # For each non looping point gets the XYZ coordniates of the closest point in the point cloud
    # Fot all points where the was no point in the point cloud within the cone, sets the XYZ coordinates to nan
    pc_per_masked_face = point_cloud[arg_min, :].copy()
    pc_per_masked_face[min == float("inf"), :] = float("nan")

    # creates one of the output objects as a tensor with shape (num_faces, 3)
    pc_per_face = np.zeros(
        shape=(np.shape(mid_points)[0], 3), dtype=pc_per_masked_face.dtype
    )

    # for every non looping point, we have already computed the xyz point on the PC that is closest
    # so this just writes those values to be the percomputed values
    pc_per_face[~looping_mesh_points_mask, :] = pc_per_masked_face

    # for all the looping points, just set the closest poinct cloud XYZs to nan
    pc_per_face[looping_mesh_points_mask, :] = float("nan")

    # checks to see if the XYZ points are nan becasue nan=/=nan
    #   True indicates the XYZ point is real which happens when the corresponing mesh midpoint is non-looping AND
    #       there is a point on the point cloud within the cone
    pc_is_non_nan = pc_per_face[:, 0] == pc_per_face[:, 0]

    # return objects
    return tf.convert_to_tensor(pc_per_face), tf.convert_to_tensor(pc_is_non_nan)


def get_looping_points(mid_points, point_cloud, k=3):
    """
    ARGS:
        mid_points: numpy array of shape (num_faces, 3) representing the XYZ locations of the center of a mesh's faces
        point_cloud: numpy array of shape (num_points_in_cloud, 3) representing the XYZ locations of the points in the cloud

    return:
        looping_mesh_points_mask: numpy boolean array of shape (num_faces) that is True if any of the k nearest points cloud
            points of the row index's mesh point has the original mesh point amoung its k nearest neighbors
    """

    # create a scipy KDTree to represent points
    # ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html
    KDTree_cloud = scipy.spatial.KDTree(point_cloud)
    KDTree_mesh = scipy.spatial.KDTree(mid_points)

    # querries each KDTree to find the k nearest neighbors. As an example:
    # k_closest_cloud_points2mesh will have the shape (np.shape(mid_points)[0], k (in our case 3))
    # each row corresponds to a midpoint, the nth index in that row corresponds to the index row into point_cloud
    # that is nth closest to the given midpoint
    # The same is querried for the mesh cloud as well (can be thought of as bi directional)
    _, k_closest_cloud_points2mesh = KDTree_cloud.query(
        mid_points, k=k, eps=0
    )  # (num_faces, k)
    _, k_closest_mesh_points2cloud = KDTree_mesh.query(
        point_cloud, k=k, eps=0
    )  # (num_points_in_cloud, k)

    # gets, for every face mid_point in the mesh, the k nearest neighbors in the point cloud
    # then, for all k of those points in the point cloud, gets the k nearest points in the set of mid_points
    # finally, reshapes from (num_faces, 3, 3) to (num_faces, 9)

    # as an example if the meshes were identical (mid_points == point_cloud)
    # with points [0,1,2] in mid_points and [a,b,c] in point_cloud
    # where dist(0,1)<dist(0,2)<dist(1,2)
    # then neighbors_neighbors would apear as follows:
    # [
    # [0, 1, 2, 1, 0, 2, 2, 0, 1]
    # [1, 0, 2, 0, 1, 2, 2, 0, 1]
    # [2, 0, 1, 0, 1, 2, 1, 0, 2]
    # ]
    neighbors_neighbors = np.reshape(
        k_closest_mesh_points2cloud[k_closest_cloud_points2mesh],
        (np.shape(mid_points)[0], -1),
    )  # (num_faces, 9)

    # essentaially this checks to see if there are any adjacency loops
    # neighbors_neighbors == arange checks to see, for each mind_point n, are its neighbor's neighbors n?
    #   True if mapping back to itself, False if not.
    #   boolean array of shape (num_faces, 9)
    # sum along axis 1 to see how many times a point maps back to itself
    #   m number of maps back to itself, max(m) = 3
    #   int array of shape (num_faces)
    # check if remaps are > 0:
    #   True if a point maps back to itself at least once amoung 9 options
    #   boolean array of shape (num_faces)

    looping_mesh_points_mask = (
        np.sum(
            np.equal(
                neighbors_neighbors,
                np.expand_dims(np.arange(0, np.shape(mid_points)[0]), axis=1),
            ),
            axis=1,
        )
        > 0
    )

    return looping_mesh_points_mask


def distance_within_cone(mid_points, normals, point_cloud, threshold):
    """
    ARGS:
        mid_points: numpy array of shape (num_faces, 3) representing the XYZ locations of the center of a mesh's faces
        normals: numpy array of shape (num_faces, 3) representing the normal vector to the face
        point_cloud: numpy array of shape (num_points_in_cloud, 3) representing the XYZ locations of the points in the cloud
        threshold: a float representing the cosine of the cone angle when checking to see if there exist any points
            within the target pointcloud that are relevant for the mash

    return:
        distance: numpy arrar of shape (num_faces, num_points) which is the l2 distance from each of the face points to each of the
            point cloud points
        masked_within_cone: numpy boolean array of shape (num_faces, num_points) that is True if the (unit displacement between
            the face mid_point and the point cloud point) dotted with (the face unit normal) is absolute value greater than the
            threshold
    """

    # This uses broadcasting to compute the displacement from every point in the mid_points
    # to every point in the point cloud
    displacement = (
        mid_points[:, None, :] - point_cloud
    )  # (num_mid_points, num_points_in_cloud, 3)

    # Compute distance via L2 norm on displacements
    distance = np.linalg.norm(
        displacement, axis=-1
    )  # (num_mid_points, num_points_in_cloud)

    # Another big, multistep operation.
    # First, normalize the displacement by its distance to compute the unit normal in the direction of the displacement
    # Second, compute the dot product of this norm with the mesh surface norm:
    #   If the dot product is close to 1 then these vectors are nearly parallel
    #   If the dot product is close to 0 then these vectors are perpendicular
    # Third, create the mask by checking if the dot product is close to 1 (within threshold)
    # In the end, boolean array of shape (num_non_looping)
    #   True if there is a point within the cone

    masked_within_cone = np.array(
        np.abs(
            np.sum(
                (displacement / distance[:, :, None]) * normals[:, None, :],
                axis=-1,
            )
        )
        > threshold
    )

    return distance, masked_within_cone