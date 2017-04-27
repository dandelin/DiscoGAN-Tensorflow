class batch_norm(object):
    def __init__(self, epsilon = 0.001, momentum = 0.99, name = "batch_norm"):
        self.epsilon = epsilon
        self.momentum = momentum
        self.name = name

    def __call__(self, x, phase = None):   # If phase == True, then training phase. If phase == False, then test phase.
        tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,  #The more you have your data, the bigger the momentum has to be
                      updates_collections=None, 
                      epsilon=self.epsilon,
                      scale=True,  # If next layer is linear like "Relu", this can be set as "False". Because You don't really need gamma in this case.
                      is_training=phase,  # Training mode or Test mode 
                      scope=self.name)




class conv_information(object):
    
    def __init__(self, conv_infos):
        self.conv_layer_number = conv_infos["conv_layer_number"]
        self.filter = conv_infos["filter"]
        self.stride = conv_infos["stride"]
        self.current = 0
 
    
    def __iter__(self):
        return self
    
    def __next__(self):
        
        if self.current >= self.conv_layer_number:
            raise StopIteration
        else:
            self.current += 1
            return self

def lrelu(tensor, alpha=0.01):
    return tf.maximum(alpha * tensor, tensor)