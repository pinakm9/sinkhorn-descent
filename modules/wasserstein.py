import tensorflow as tf
tf.keras.backend.set_floatx('float32')

def cost_matrix(x, y, p=2):
    "Returns the cost matrix C_{ij}=|x_i - y_j|^p"
    x_col = tf.expand_dims(x,1)
    y_lin = tf.expand_dims(y,0)
    c = tf.reduce_sum((tf.abs(x_col-y_lin))**p,axis=2)
    return c

def tf_round(x, decimals):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier

#@tf.function
def sinkhorn_div_tf(x, y, alpha=None, beta=None, epsilon=0.01, num_iters=200, p=2):
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    c = cost_matrix(x, y, p=p) 

    if alpha is None:
        alpha = tf.ones(x.shape[0], dtype=x.dtype)
    else:
        alpha = tf.convert_to_tensor(alpha)
    alpha /= tf.reduce_sum(alpha)

    if beta is None:
        beta = tf.ones(y.shape[0], dtype=y.dtype)
    else:
        beta = tf.convert_to_tensor(beta)
    beta /= tf.reduce_sum(beta)

    log_alpha = tf.expand_dims(tf.math.log(alpha), 1)
    log_beta = tf.math.log(beta)

    f, g = 0. * alpha, 0. * beta
    f_, iter = 1. * alpha, 0
    
    while (tf.norm(f - f_, ord=1) / tf.norm(f_, ord=1) > 1e-3) and iter < num_iters:
        f_ = 1.0 * f
        f = 0.- epsilon * tf.reduce_logsumexp(log_beta + (g - c) / epsilon, axis=1)
        g = - epsilon * tf.reduce_logsumexp(log_alpha + (tf.expand_dims(f, 1) - c) / epsilon, axis=0)
        iter += 1
    #print('iteration count = {}'.format(iter))

    OT_alpha_beta = tf.reduce_sum(f * alpha) + tf.reduce_sum(g * beta)
    
    c = cost_matrix(x, x, p=p)
    f_ = 0. * alpha
    f__, iter = 1. * alpha, 0
    log_alpha = tf.squeeze(log_alpha)
    while tf.norm(f_ - f__, ord=1) / tf.norm(f__, ord=1) > 1e-3 and iter < num_iters:
        f__ = 1.0 * f_
        f_ = 0.5 * (f_ - epsilon * tf.reduce_logsumexp(log_alpha + (f_ - c) / epsilon, axis=1) )
        iter += 1
    #print(iter)

    c = cost_matrix(y, y, p=p)
    g_ = 0. * beta
    g__, iter = 1. * beta, 0
    while tf.norm(g_ - g__, ord=1) / tf.norm(g__, ord=1) > 1e-3 and iter < num_iters:
        g__ = 1.0 * g_
        g_ = 0.5 * (g_ - epsilon * tf.reduce_logsumexp(log_beta + (g_ - c) / epsilon, axis=1) )
        iter += 1
    
    sd = tf_round(OT_alpha_beta - tf.reduce_sum(f_ * alpha) - tf.reduce_sum(g_ * beta), 5)
    #print(d**0.5)
    return sd, f, g, f_, g_ #tf_round(OT_alpha_beta - tf.reduce_sum(f * alpha) - tf.reduce_sum(g * beta), 5)

@tf.function
def sink_grad_position(x, y, alpha=None, beta=None, epsilon=0.01, num_iters=200, p=2):
    sd, f, g, f_, g_ = sinkhorn_div_tf(x, y, alpha, beta, epsilon, num_iters, p)
    log_alpha, log_beta = tf.math.log(alpha), tf.math.log(beta)
    grads = []
    def phi(z):
        #z = tf.concat(args, axis=0)
        c = tf.reduce_sum((tf.abs(z - y))**p, axis=1)
        a = - epsilon * tf.reduce_logsumexp(log_beta + (g - c) / epsilon) 
        c_ = tf.reduce_sum((tf.abs(z - x))**p, axis=1)
        b = epsilon * tf.reduce_logsumexp(log_alpha + (f_ - c_) / epsilon)
        return a + b
    grads = []
    for z in x:
        with tf.GradientTape() as tape:
            tape.watch(z)
            f = phi(z)
        df = tape.gradient(f, z)
        grads.append(df)
    return sd, grads


class UniformSampleFinder():
    """
    Class that finds a uniform sample given a weighted sample
    """
    def __init__(self, w_sample, weights, epsilon=0.01, n_sink_iters=50, cost_p=2):
        self.w_sample = w_sample
        self.weights = weights
        self.epsilon = epsilon 
        self.n_sink_iters = n_sink_iters 
        self.cost_p = cost_p
        self.u_sample = [tf.Variable for s in w_sample]

    def find(self, n_iters=100, learning_rate=1e-2):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        for iter in range(n_iters):
            sd, grads = sink_grad_position(self.u_sample, self.w_sample, None, self.weights,\
                        self.epsilon, self.n_sink_iters, self.cost_p)
            print('Step = {}, Sinkhorn divergence = {}'.format(iter+1, sd.numpy()), end='\n')
            optimizer.apply_gradients(zip(grads, self.u_sample))
        return self.u_sample
        

