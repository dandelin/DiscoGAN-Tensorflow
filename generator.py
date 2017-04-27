import tensorflow as tf
from util import lrelu, batch_norm

class Generator(object):
    def __init__(
        self, conv_infos, deconv_infos,
        in_dim=(64, 64), out_dim=(64, 64), channel=3, signature='AB'
    ):
        
        self.in_dim = in_dim
        self.a_height = in_dim[0]
        self.a_width = in_dim[1]
        
        self.out_dim = out_dim
        self.b_height = out_dim[0]
        self.b_width = out_dim[1]
        
        self.channel = channel
        self.signature = signature

        self.conv_infos = conv_infos
        self.deconv_infos = deconv_infos
    

    def build_model(self, image, reuse=False):
        with tf.variable_scope('is_training', reuse=True):
            is_training = tf.get_variable('is_training', dtype=tf.bool)
        with tf.variable_scope("G_" + self.signature) as scope:
            if reuse:
                scope.reuse_variables()
            
            prev = image

            for i in range(self.conv_infos['conv_layer_number']):
                weight = tf.get_variable('conv_weight_' + str(i), shape=self.conv_infos['filter'][i], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                setattr(self, "conv_weight_" + str(i), weight)

                conv = tf.nn.conv2d(prev, weight, self.conv_infos['stride'][i], padding="SAME", name="g_conv_" + str(i))
                setattr(self, "conv_" + str(i), conv)

                bn = batch_norm(name='conv_bn_' + str(i))
                setattr(self, "conv_bn_" + str(i), bn) 

                normalized_layer = bn(conv, phase=is_training)      # arg "phase" has to be specified whether it is training or test session         
                activated_conv = lrelu(normalized_layer) #Right after conv layer, relu function has not yet specified.
                prev = activated_conv

            for i in range(self.deconv_infos['conv_layer_number']):
                weight = tf.get_variable('deconv_weight_' + str(i), shape=self.deconv_infos['filter'][i], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                setattr(self, "deconv_weight_" + str(i), weight)

                conv_t = tf.nn.conv2d_transpose(prev, weight, self.deconv_infos['output_dims'][i], self.deconv_infos['stride'][i], padding="SAME", name="g_deconv_" + str(i))
                if i == self.deconv_infos['conv_layer_number'] - 1:
                    return tf.sigmoid(conv_t)
                bn = batch_norm(name='deconv_bn_' + str(i))
                setattr(self, "deconv_bn_" + str(i), bn) 

                normalized_layer = bn(conv_t, phase=is_training)              # arg "phase" has to be specified whether it is training or test session  
                activated_conv = tf.nn.relu(normalized_layer) # Right after conv layer, relu function has not yet specified.
                prev = conv_t