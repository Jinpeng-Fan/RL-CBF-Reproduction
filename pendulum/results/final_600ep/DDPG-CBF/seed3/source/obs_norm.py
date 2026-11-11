"""Running observation normalization (as VecNormalize in Stable-Baselines3).

x -> clip((x - mean) / sqrt(var + eps), -clip, clip), applied inside the
TF graph, so every network still receives raw observations and GP/CBF keep
using raw physical states. Statistics are non-trainable TF variables: they
are saved with checkpoints and never appear in tf.trainable_variables().
One normalizer is shared by all networks of an agent.
"""
import numpy as np
import tensorflow as tf


class ObsNormalizer():
    def __init__(self, sess, size, clip=10., epsilon=1e-8, name='ObsNorm'):
        self.sess, self.size, self.clip, self.epsilon = sess, size, clip, epsilon
        with tf.variable_scope(name):
            self.mean = tf.Variable(np.zeros(size, np.float32), trainable=False, name='mean')
            self.var = tf.Variable(np.ones(size, np.float32), trainable=False, name='var')
            self.count = tf.Variable(1e-4, dtype=tf.float32, trainable=False, name='count')
            self.new_mean = tf.placeholder(tf.float32, [size])
            self.new_var = tf.placeholder(tf.float32, [size])
            self.new_count = tf.placeholder(tf.float32, [])
            self.assign = tf.group(self.mean.assign(self.new_mean), self.var.assign(self.new_var),
                                   self.count.assign(self.new_count))

    def __call__(self, x):
        """Normalized tensor for a raw observation tensor x."""
        return tf.clip_by_value((x - self.mean) / tf.sqrt(self.var + self.epsilon),
                                -self.clip, self.clip)

    def update(self, batch):
        """Merge a batch of raw observations into the running statistics."""
        batch = np.asarray(batch, np.float64).reshape(-1, self.size)
        mean, var, count = [np.asarray(v, np.float64) for v in
                            self.sess.run([self.mean, self.var, self.count])]
        batch_mean, batch_var, batch_count = batch.mean(0), batch.var(0), len(batch)
        delta = batch_mean - mean
        total = count + batch_count
        new_mean = mean + delta * batch_count / total
        m2 = var * count + batch_var * batch_count + delta ** 2 * count * batch_count / total
        self.sess.run(self.assign, {self.new_mean: new_mean, self.new_var: m2 / total,
                                    self.new_count: total})
