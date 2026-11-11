import numpy as np
import tensorflow as tf
tf1 = tf.compat.v1


#Initialize or run neural network layer
def LINEAR(x, hidden, name='None', weight_std=0.02, bias_std=0.1):
    with tf1.variable_scope(name, reuse=tf1.AUTO_REUSE):
        weight = tf1.get_variable('Weight', [x.get_shape()[-1], hidden],
                                  initializer=tf1.truncated_normal_initializer(stddev=weight_std))  # Near empty NN
        if bias_std > 0:
            bias_init = tf1.truncated_normal_initializer(stddev=bias_std)
        else:
            bias_init = tf1.constant_initializer(0.)
        bias = tf1.get_variable('Bias', [hidden, ], initializer=bias_init)
        return tf.matmul(x, weight) + bias


#Calculate probability of action (mu, logstd : [batch size, action size])
def LOG_POLICY(mu, logstd, action):
    variance = tf.exp(2 * logstd)
    log_prob = -tf.square(action - mu) / (2 * variance) - 0.5 * tf.math.log(2 * np.pi) - logstd
    return tf.reduce_sum(log_prob, 1)


#KL divergence between two parameterized Gaussians
def GAUSS_KL(mu1, logstd1, mu2, logstd2):
    variance1 = tf.exp(2 * logstd1)
    variance2 = tf.exp(2 * logstd2)
    kl = logstd2 - logstd1 + (variance1 + tf.square(mu1 - mu2)) / (2 * variance2) - 0.5
    return tf.reduce_sum(kl)


#Shannon entropy of gaussian
def GAUSS_ENTROPY(mu, logstd):
    variance = tf.exp(2 * logstd)
    entropy = (1 + tf.math.log(2 * np.pi * variance)) / 2
    return tf.reduce_sum(entropy)


#Flatten gradient along all variables
def FLAT_GRAD(loss, vrbs):
    grads = tf1.gradients(loss, vrbs)
    return tf.concat(values=[tf.reshape(g, [int(np.prod(v.get_shape().as_list())), ]) for (g, v) in zip(grads, vrbs)], axis=0)


def GAUSS_KL_FIRST_FIX(mu, logstd):
    #First argument is old policy, so keep it unchanged through tf.stop_gradient
    mu1, logstd1 = map(tf.stop_gradient, [mu, logstd])
    mu2, logstd2 = mu, logstd
    return GAUSS_KL(mu1, logstd1, mu2, logstd2)


# Get actual parameter values
class GetValue:
    def __init__(self, sess, variable_list, name=None):
        self.name = name
        self.sess = sess
        self.op_list = tf.concat(axis=0, values=[tf.reshape(v, [int(np.prod(v.get_shape().as_list()))]) for v in variable_list])

    def __call__(self):
        return self.sess.run(self.op_list)


# Set parameter values
class SetValue:
    def __init__(self, sess, variable_list, name=None):
        self.name = name
        self.sess = sess
        shape_list = [v.get_shape().as_list() for v in variable_list]
        total_variable_size = int(sum(int(np.prod(s)) for s in shape_list))
        print('Total variable size (%s): %d' % (name, total_variable_size))

        self.var_list = var_list = tf1.placeholder(tf.float32, [total_variable_size])
        start = 0
        assign_ops = list()
        for (shape, var) in zip(shape_list, variable_list):
            variable_size = int(np.prod(shape))
            assign_ops.append(tf1.assign(var, tf.reshape(var_list[start:(start + variable_size)], shape)))
            start += variable_size
        self.op_list = tf.group(*assign_ops)

    def __call__(self, var, update_info=0):
        self.sess.run(self.op_list, feed_dict={self.var_list: var})


# Take Hessian product with y (w.r.t. variables vrbs)
def HESSIAN_VECTOR_PRODUCT(func, vrbs, y):
    first_derivative = tf1.gradients(func, vrbs)
    flat_y = list()
    start = 0
    for var in vrbs:
        variable_size = int(np.prod(var.get_shape().as_list()))
        param = tf.reshape(y[start:(start + variable_size)], var.get_shape())
        flat_y.append(param)
        start += variable_size
    gradient_with_y = [tf.reduce_sum(f_d * f_y) for (f_d, f_y) in zip(first_derivative, flat_y)]
    return FLAT_GRAD(gradient_with_y, vrbs)


def DISCOUNT_SUM(x, discount_factor):
    """[x1 + df*x2 + df^2*x3 + ..., x2 + df*x3 + ..., ...]  (1-D input, one episode)."""
    x = np.asarray(x, dtype=float).reshape(-1)
    out = np.zeros_like(x)
    running = 0.0
    for i in range(len(x) - 1, -1, -1):
        running = x[i] + discount_factor * running
        out[i] = running
    return out


def CONJUGATE_GRADIENT(fvp, y, k=10, tolerance=1e-6):
    p = y.copy()
    r = y.copy()
    x = np.zeros_like(y)
    r_transpose_r = r.dot(r)
    for i in range(k):
        FIM_p = fvp(p)
        denom = p.dot(FIM_p)
        if not np.isfinite(denom) or abs(denom) < 1e-30:
            break
        alpha_k = r_transpose_r / denom
        x += alpha_k * p
        r -= alpha_k * FIM_p
        new_r_transpose_r = r.dot(r)
        beta_k = new_r_transpose_r / r_transpose_r
        p = r + beta_k * p
        r_transpose_r = new_r_transpose_r
        if r_transpose_r < tolerance:
            break
    return x


def LINE_SEARCH(surr, theta_prev, full_step, num_backtracking=10, name=None):
    """Backtracking: accept the first step that decreases surr(theta)."""
    prev_sur_objective = surr(theta_prev)
    for num_bt, fraction in enumerate(0.5 ** np.arange(num_backtracking)):
        theta_new = theta_prev + full_step * fraction
        if prev_sur_objective - surr(theta_new) > 0:
            return theta_new
    return theta_prev


def TRUST_REGION_STEP(evaluate, theta_prev, gradient, fvp, kl_limit, num_backtracking=20):
    """Accept only finite surrogate improvement AND measured mean KL <= limit.

    evaluate(theta) assigns theta and returns (loss, empirical KL) on the same saved rollout.
    Approximate Fisher curvature only proposes a step.
    """
    if not np.isfinite(kl_limit) or kl_limit <= 0:
        raise ValueError('KL limit must be finite and positive.')
    old_loss, old_kl = evaluate(theta_prev)
    if not np.isfinite(old_loss) or not np.isfinite(old_kl):
        raise FloatingPointError('Nonfinite old-policy objective or KL.')
    info = dict(accepted=False, backtracks=0, status='invalid_direction')
    direction = CONJUGATE_GRADIENT(fvp, -gradient)
    curvature = 0.5 * float(direction.dot(fvp(direction)))
    if np.isfinite(curvature) and curvature > 0 and np.all(np.isfinite(direction)):
        full_step = np.sqrt(kl_limit / curvature) * direction
        if np.all(np.isfinite(full_step)):
            info['status'] = 'line_search_rejected'
            for j, fraction in enumerate(0.5 ** np.arange(num_backtracking)):
                candidate = theta_prev + fraction * full_step
                if not np.all(np.isfinite(candidate)):
                    continue
                loss, kl = evaluate(candidate)
                info['backtracks'] = j
                if np.isfinite(loss) and np.isfinite(kl) and loss < old_loss and -1e-6 <= kl <= kl_limit:
                    info.update(accepted=True, status='accepted', kl=float(kl))
                    return candidate, info
    evaluate(theta_prev)
    info['kl'] = float(old_kl)
    return theta_prev.copy(), info
