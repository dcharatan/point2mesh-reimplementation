import tensorflow as tf
import tensorflow_graphics.nn.loss.chamfer_distance as chamfer_distance
from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp


class ChamferLossLayer(Layer):
    def __init__(self, max_num_samples=20000) -> None:
        super(ChamferLossLayer, self).__init__()
        self.max_num_samples = max_num_samples

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

        # Assert that cloud1 and cloud2 are tensors of the correct shape.
        assert tf.is_tensor(cloud1)
        assert tf.shape(cloud1)[-1] == 3
        assert tf.is_tensor(cloud2)
        assert tf.shape(cloud2)[-1] == 3

        if tf.shape(cloud1)[-2] > self.max_num_samples:
            num_points = tf.shape(cloud1)[-2]
            test = tf.ones(num_points)
            point_sample_probs = test / tf.cast(num_points, dtype=tf.float32)
            point_distribution = tfp.distributions.Categorical(probs=point_sample_probs)
            points_to_sample = point_distribution.sample(self.max_num_samples)
            cloud1 = tf.gather(cloud1, points_to_sample)

        if tf.shape(cloud2)[-2] > self.max_num_samples:
            num_points = tf.shape(cloud2)[-2]
            test = tf.ones(num_points)
            point_sample_probs = test / tf.cast(num_points, dtype=tf.float32)
            point_distribution = tfp.distributions.Categorical(probs=point_sample_probs)
            points_to_sample = point_distribution.sample(self.max_num_samples)
            cloud2 = tf.gather(cloud2, points_to_sample)

        # Compute bidirectional, average loss using built in tfg function.
        # Returns the sum of (the mean, minimum, squared distance from cloud 1
        # to 2) and vice-versa (the mean, minimum, squared distance from cloud 2
        # to 1).
        return chamfer_distance.evaluate(cloud1, cloud2)