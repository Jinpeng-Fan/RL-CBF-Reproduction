"""Fixed physical scaling for pendulum network inputs.

The policy, value network and compensator share [cos(theta), sin(theta),
theta_dot / (8 rad/s)]. There are no running statistics to update between
batches. GP and CBF calculations continue to receive raw physical states.
"""
import tensorflow as tf


class FixedObservationScaler():
    def __init__(self, size, clip=10., name='ObsNorm'):
        if size != 3:
            raise ValueError('Pendulum scaling expects [cos(theta), sin(theta), theta_dot].')
        self.clip = clip
        with tf.name_scope(name):
            self.scale = tf.constant([1., 1., 8.], dtype=tf.float32, name='scale')

    def __call__(self, x):
        # Keep the existing clipping bound; only the normalization is now fixed.
        return tf.clip_by_value(x / self.scale, -self.clip, self.clip)
