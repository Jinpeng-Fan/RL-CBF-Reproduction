"""Fixed physical scaling of the 15-dim observation, shared by the policy,
the value network and the compensator (no running statistics).

Absolute positions are replaced by the four neighbour gaps:
  [gap12, gap23, gap34, gap45]/10, [v_i - 30]/5 (5 values), a_car4/10  -> 10 features
"""
import tensorflow as tf


class FixedObservationScaler():
    def __init__(self, size=15, clip=10., name='ObsNorm'):
        if size != 15:
            raise ValueError('Car scaling expects 15 observations.')
        self.clip = clip
        self.out_size = 10
        self.name = name

    def __call__(self, x):
        with tf.name_scope(self.name):
            p, v, a = x[:, 0:15:3], x[:, 1:15:3], x[:, 2:15:3]
            gaps = (p[:, :-1] - p[:, 1:]) / 10.
            vel = (v - 30.) / 5.
            acc4 = a[:, 3:4] / 10.
            feat = tf.concat([gaps, vel, acc4], axis=1)
            return tf.clip_by_value(feat, -self.clip, self.clip)
