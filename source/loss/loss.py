from tensorflow.python.keras.backend import dtype
from source.mesh.Mesh import discrete_project
import numpy as np
import tensorflow as tf
import tensorflow_graphics as tfg
from typing import Union
from tensorflow.keras.layers import Layer


class ChamferLossLayer(Layer):
    def __init__(self) -> None:
        super(ChamferLossLayer, self).__init__()

    def call(self, cloud1, cloud2):
        """
        Arguments:
        cloud1 is a tf tensor of shape (N,P1,D) where:
            N is number of clouds in batch,
            P1 is the max number of points amoung all of the images in the batch
            D is dimensionality of system
        cloud2 is a tf tensor of shape (M,P2,D) where:
            N is number of clouds in batch,
            P2 is the max number of points amoung all of the images in the batch
            D is dimensionality of system
        """

        # asserts that cloud1 and cloud2 are tensors of the correct shape
        assert tf.is_tensor(cloud1)
        assert tf.shape(cloud1)[-1] == 3
        assert tf.is_tensor(cloud2)
        assert tf.shape(cloud2)[-1] == 3

        # computes bidirectional, average loss using built in tfg function
        cham_distance = tfg.nn.loss.chamfer_distance.evaluate(cloud1, cloud2)

        return cham_distance


class BeamGapLossLayer(Layer):
    def __init__(self, device) -> None:

        super(BeamGapLossLayer, self).__init__()

        # Sets the device to use equal to device
        self.device = device

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
        self.points, self.mask = mesh.discrete_project(target_point_cloud, 0.99)

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
        losses = self.points - tf.math.reduce_mean(mesh.verticies[mesh.faces], axis=1)

        # compute the "length" of the loss by computing norm of the XYZ
        # also only select the non-nan points for the loss
        losses = tf.norm(losses, axis=-1)[self.mask]  # (numfaces)

        # take the reduced mean of all the lengths of the losses and cast as float32. Mutliply by 10
        l2 = 10.0 * tf.cast(tf.math.reduce_mean(losses), dtype=tf.float32)  # ()

        return l2
