import tensorflow as tf

class batch_norm(object):
    def __init__(self, epsilon = 0.001, momentum = 0.99, name = "batch_norm"):
        self.epsilon = epsilon
        self.momentum = momentum
        self.name = name

    def __call__(self, x, phase = None):   # If phase == True, then training phase. If phase == False, then test phase.
        return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,  #The more you have your data, the bigger the momentum has to be
                      updates_collections=None, 
                      epsilon=self.epsilon,
                      scale=True,  # If next layer is linear like "Relu", this can be set as "False". Because You don't really need gamma in this case.
                      is_training=phase,  # Training mode or Test mode 
                      scope=self.name)

def lrelu(tensor, alpha=0.01):
    return tf.maximum(alpha * tensor, tensor)