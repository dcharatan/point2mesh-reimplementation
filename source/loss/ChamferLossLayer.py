import tensorflow as tf
import tensorflow_graphics.nn.loss.chamfer_distance as chamfer_distance
from tensorflow.keras.layers import Layer
import tensorflow_probability as tfp


class ChamferLossLayer(Layer):
    def __init__(self) -> None:
        super(ChamferLossLayer, self).__init__()

    def call(self, cloud1, cloud2, num_samples):
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

        def subsample(cloud):
            num_points = tf.shape(cloud)[-2]
            point_sample_probs = tf.ones(num_points) / tf.cast(
                num_points, dtype=tf.float32
            )
            point_distribution = tfp.distributions.Categorical(probs=point_sample_probs)
            points_to_sample = point_distribution.sample(num_samples)
            return tf.gather(cloud, points_to_sample)

        subsampled_cloud1 = subsample(cloud1)
        subsampled_cloud2 = subsample(cloud2)

        # Compute bidirectional, average loss using built in tfg function.
        # Returns the sum of (the mean, minimum, squared distance from cloud 1
        # to 2) and vice-versa (the mean, minimum, squared distance from cloud 2
        # to 1).
        return chamfer_distance.evaluate(subsampled_cloud1, subsampled_cloud2)