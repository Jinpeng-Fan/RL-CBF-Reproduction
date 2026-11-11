import tensorflow as tf
import numpy as np

#Initialize or run neural network layer
def LINEAR(x, hidden, name='None'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        #Set zero for initial weights (i.e. start out with 'empty' NN)
        #weight = tf.get_variable('Weight', [x.get_shape()[-1], hidden], initializer=tf.constant_initializer(0))
        weight = tf.get_variable('Weight', [x.get_shape()[-1], hidden], initializer=tf.truncated_normal_initializer(stddev=0.03))  # Near empty NN
        
        #Base initializer
        #weight = tf.get_variable('Weight', [x.get_shape()[-1], hidden], initializer=tf.truncated_normal_initializer(stddev=0.02))
        bias = tf.get_variable('Bias', [hidden,], initializer=tf.constant_initializer(0))
        weighted_sum = tf.matmul(x,weight) + bias
    return weighted_sum

#Calculate probability of action (mu, logstd : [batch size, action size])
def LOG_POLICY(mu, logstd, action):
    #Get log probability of action given mean/variance
    variance = tf.exp(2*logstd)
    log_prob = -tf.square(action-mu)/(2*variance) - 0.5*tf.log(2*np.pi) - logstd

    #Sum along 'action size' axis to get product of probability of each action index
    return tf.reduce_sum(log_prob,1)

#KL divergence between two parameterized Gaussians
def GAUSS_KL(mu1, logstd1, mu2, logstd2):
    variance1 = tf.exp(2*logstd1)
    variance2 = tf.exp(2*logstd2)
    
    kl = logstd2 - logstd1 + (variance1 + tf.square(mu1 - mu2))/(2*variance2) - 0.5
    return tf.reduce_sum(kl)

#Shannon entropy of gaussian
def GAUSS_ENTROPY(mu, logstd):
    variance = tf.exp(2*logstd)
    entropy = (1 + tf.log(2*np.pi*variance))/2
    return tf.reduce_sum(entropy)

#CHECK THIS FUNCTION
#Flatten gradient along all variables
def FLAT_GRAD(loss, vrbs):
    # tf.gradients returns list of gradients w.r.t. variables
    '''
        If loss argument is list, tf.gradients returns sum of gradient of each loss element for each variable
        tf.gradients([y,z]) => [dy/dx + dz/dx] where x is variable
    '''
    grads = tf.gradients(loss, vrbs)

    #Ensures each gradient has same shape with variables    
    return tf.concat(values=[tf.reshape(g, [np.prod(v.get_shape().as_list()),]) for (g,v) in zip(grads,vrbs)], axis=0)

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
        self.op_list = tf.concat(axis=0, values=[tf.reshape(v, [np.prod(v.get_shape().as_list())]) for v in variable_list])
        
    #Use class instance as function
    def __call__ (self):
        #print('Getting %s parameter value' % self.name)
        return self.op_list.eval(session=self.sess)
    
# Set policy parameter values
class SetValue:
    def __init__(self, sess, variable_list, name=None):
        self.name = name
        self.sess = sess
        shape_list = list()
        
        # Get 'shape/size' of variable list
        for i in variable_list:
            shape_list.append(i.get_shape().as_list())
        total_variable_size = np.sum(np.prod(shapes) for shapes in shape_list)
        print('Total variable size : %d' % total_variable_size)
        
        #Assign variables in variable list
        self.var_list = var_list = tf.placeholder(tf.float32,[total_variable_size])
        start = 0
        assign_ops = list()
        for (shape, var) in zip(shape_list, variable_list):
            variable_size = np.prod(shape)
            assign_ops.append(tf.assign(var, tf.reshape(var_list[start:(start+variable_size)], shape)))
            start += variable_size
        
        self.op_list = tf.group(*assign_ops)
        
    def __call__(self, var, update_info=0):
        if update_info:
            print('Update %s parameter' % self.name)
        self.sess.run(self.op_list, feed_dict={self.var_list:var})

# Take Hessian product with y (w.r.t. variables vrbs)
def HESSIAN_VECTOR_PRODUCT(func, vrbs, y):
    first_derivative = tf.gradients(func, vrbs)
    flat_y = list()
    start = 0
    for var in vrbs:
        variable_size = np.prod(var.get_shape().as_list())
        param = tf.reshape(y[start:(start+variable_size)], var.get_shape())
        flat_y.append(param)
        start += variable_size
    #First derivative * y
    gradient_with_y = [tf.reduce_sum(f_d * f_y) for (f_d, f_y) in zip(first_derivative, flat_y)]
    HVP = FLAT_GRAD(gradient_with_y, vrbs)
    return HVP

'''
    'x' should be array of shape [batch size, ]
    Return: [x1 + df*x2 + x3*df**2 + ..., x2 + df*x3 + (df**2)*x4 + ..., x3 + ..., ...]
'''
def DISCOUNT_SUM(x, discount_factor, print_info=None):
    size = x.shape[0]
    if print_info is not None:
        print('Input shape', size, 'Discount factor', discount_factor)
    discount_sum = np.zeros((size,))
    # x[::-1] is reverse of x
    for idx, value in enumerate(x[::-1]):
        discount_sum[:size-idx] += value
        if size-idx-1 == 0:
            break
        discount_sum[:size-idx-1] *= discount_factor
        
    return discount_sum
    
def CONJUGATE_GRADIENT(fvp, y, k=10, tolerance=1e-6):
    """Truncated CG: never divide by zero or follow nonpositive curvature."""
    y = np.asarray(y)
    x = np.zeros_like(y)
    if not np.all(np.isfinite(y)):
        return x
    r, p = y.copy(), y.copy()
    rr = float(r.dot(r))
    for _ in range(k):
        if rr <= tolerance:
            break
        Ap = np.asarray(fvp(p))
        curvature = float(p.dot(Ap))
        if not np.all(np.isfinite(Ap)) or not np.isfinite(curvature) or curvature <= 0:
            break
        alpha = rr / curvature
        xn, rn = x + alpha*p, r - alpha*Ap
        rrn = float(rn.dot(rn))
        if not np.all(np.isfinite(xn)) or not np.isfinite(rrn):
            break
        x, r = xn, rn
        p = r + (rrn/rr)*p
        rr = rrn
    return x

    
def LINE_SEARCH(surr, theta_prev, full_step, num_backtracking=10, name=None):
    """Finite decreasing step for value/compensation objectives (not TRPO)."""
    previous = surr(theta_prev)
    if np.isfinite(previous) and np.all(np.isfinite(full_step)):
        for fraction in 0.5**np.arange(num_backtracking):
            candidate = theta_prev + fraction*full_step
            if not np.all(np.isfinite(candidate)):
                continue
            objective = surr(candidate)
            if np.isfinite(objective) and objective < previous:
                return candidate
    surr(theta_prev)  # Restore callback-owned network as well as returned vector.
    return theta_prev.copy()


def TRUST_REGION_STEP(evaluate, theta_prev, gradient, fvp, kl_limit, num_backtracking=20):
    """Accept only finite surrogate improvement AND measured mean KL <= limit.

    evaluate(theta) assigns theta and returns (loss, empirical KL) on the
    same saved rollout. Approximate Fisher curvature only proposes a step.
    """
    if not np.isfinite(kl_limit) or kl_limit <= 0:
        raise ValueError('KL limit must be finite and positive.')
    old_loss, old_kl = evaluate(theta_prev)
    if not np.isfinite(old_loss) or not np.isfinite(old_kl):
        raise FloatingPointError('Nonfinite old-policy objective or KL.')
    info = dict(accepted=False, backtracks=0, status='invalid_direction')
    direction = CONJUGATE_GRADIENT(fvp, -gradient)
    curvature = 0.5*float(direction.dot(fvp(direction)))
    if np.isfinite(curvature) and curvature > 0 and np.all(np.isfinite(direction)):
        full_step = np.sqrt(kl_limit/curvature)*direction
        if np.all(np.isfinite(full_step)):
            info['status'] = 'line_search_rejected'
            for j, fraction in enumerate(0.5**np.arange(num_backtracking)):
                candidate = theta_prev + fraction*full_step
                if not np.all(np.isfinite(candidate)):
                    continue
                loss, kl = evaluate(candidate)
                info['backtracks'] = j
                if (np.isfinite(loss) and np.isfinite(kl) and loss < old_loss
                        and -1e-6 <= kl <= kl_limit):
                    info.update(accepted=True, status='accepted', kl=float(kl))
                    return candidate, info
    evaluate(theta_prev)
    info['kl'] = float(old_kl)
    return theta_prev.copy(), info

    


def VALUE_DESCENT_STEP(loss, previous, gradient, hvp, scale, num_backtracking=30):
    """Value-only Armijo search; fall back to -gradient if Newton search fails."""
    previous = np.asarray(previous).copy()
    gradient = np.asarray(gradient)
    old_loss = float(loss(previous))
    info = dict(accepted=False, method='none', backtracks=0,
                loss_before=old_loss, loss_after=old_loss, status='no_accepted_step')
    if not np.isfinite(old_loss) or not np.all(np.isfinite(gradient)):
        info['status'] = 'nonfinite_input'
        loss(previous)
        return previous, info
    if not np.any(gradient):
        info['status'] = 'zero_gradient'
        loss(previous)
        return previous, info
    direction = CONJUGATE_GRADIENT(hvp, -gradient)
    candidates = []
    if np.all(np.isfinite(direction)) and float(gradient.dot(direction)) < 0:
        curvature = 0.5 * float(direction.dot(hvp(direction)))
        if np.isfinite(curvature) and curvature > 0 and np.isfinite(scale) and scale > 0:
            candidates.append(('cg', np.sqrt(scale / curvature) * direction))
    candidates.append(('gradient_fallback', -gradient))
    for method, step in candidates:
        slope = float(gradient.dot(step))
        if not np.all(np.isfinite(step)) or not np.isfinite(slope) or slope >= 0:
            continue
        for j in range(num_backtracking):
            fraction = 0.5 ** j
            candidate = previous + fraction * step
            if not np.all(np.isfinite(candidate)):
                continue
            value = float(loss(candidate))
            info['backtracks'] += 1
            if np.isfinite(value) and value < old_loss and value <= old_loss + 1e-4 * fraction * slope:
                info.update(accepted=True, method=method, loss_after=value, status='accepted')
                return candidate, info
    loss(previous)  # Restore callback-owned parameters, including after NaN trials.
    return previous, info
