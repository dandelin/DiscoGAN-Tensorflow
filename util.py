import tensorflow as tf

class batch_norm(object):
    def __init__(self, epsilon=1e-3, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name
    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, is_training=train, scope=self.name)

def lrelu(tensor, alpha=0.2):
    return tf.maximum(alpha * tensor, tensor)

def conv2d(x, w, stride=2, padding='SAME'):
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding=padding)

def conv2d_t(x, w, shape, stride=2, padding='SAME'):
    return tf.nn.conv2d_transpose(x, w, shape, strides=[1, stride, stride, 1], padding=padding)

def weight_var(shape, name='weight', init=tf.truncated_normal_initializer(stddev=0.02)):
    return tf.get_variable(name, shape, initializer=init)

def conv_layer(x, w_shape, name, activation=tf.nn.relu, batch_norm=None, stride=2, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse: scope.reuse_variables()
        w = weight_var(w_shape, name='weight_{}'.format(name))
        h = conv2d(x, w, stride=stride)
        if batch_norm is not None:
            h = batch_norm(h)
        if activation:
            h = activation(h)
        return h

def conv_layer_t(x, w_shape, name, shape, activation=tf.nn.relu, batch_norm=None, stride=2, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse: scope.reuse_variables()
        w = weight_var(w_shape, name='weight_{}'.format(name))
        h = conv2d_t(x, w, shape)
        if batch_norm is not None:
            h = batch_norm(h)
        if activation:
            h = activation(h)
        return h

def ReconstructionLoss(x, y, method='MSE'):
    if method == 'MSE':
        return tf.losses.mean_squared_error(labels=x, predictions=y)
    elif method == 'L1':
        return tf.losses.absolute_difference(labels=x, predictions=y)

def GANLoss(logits, is_real=True, smoothing=0.9):
    if is_real:
        labels = tf.fill(logits.get_shape(), smoothing)
    else:
        labels = tf.zeros_like(logits)
    
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))