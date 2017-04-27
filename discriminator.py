import tensorflow as tf
from util import lrelu, batch_norm

class Discriminator(object):
    """
    Arguments :
        dim : (height, width) information
        channel : 3 (RGB), 1 (Greyscale, Spectrogram)
        conv_infos : convolutional layer's informations
        signature : Instance's name like D_A
    """
    def __init__(self, conv_infos, dim=(64, 64), channel=3, signature=None):
        self.dim = dim
        self.channel = channel
        self.signature = signature
        self.conv_infos = conv_infos

    def build_model(self, image, reuse=False):
        """ TO DO """
        with tf.variable_scope('is_training', reuse=True):
            is_training = tf.get_variable('is_training', dtype=tf.bool)

        with tf.variable_scope("D_" + self.signature) as scope:
            if reuse:
                scope.reuse_variables()

            prev = image

            for i in range(self.conv_infos['conv_layer_number']):
                """
                    filter = 4 x 4, strides = [1, stride, stride, 1]
                    add batch normalization
                    add tensorboard summaries
                """
                weight = tf.get_variable('conv_weight_' + str(i), shape=self.conv_infos['filter'][i], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                setattr(self, "conv_weight_" + str(i), weight)

                conv = tf.nn.conv2d(prev, weight, self.conv_infos['stride'][i], padding="SAME", name="d_conv_" + str(i))
                setattr(self, "conv_" + str(i), conv)

                if i == self.conv_infos['conv_layer_number'] - 1:
                    # no batch_norm and leaky relu activation at the last layer
                    return tf.sigmoid(conv)

                else:
                    bn = batch_norm(name='d_bn_' + str(i))
                    setattr(self, "bn_" + str(i), bn)

                    # arg "phase" has to be specified whether it is training or test session
                    normalized_layer = bn(conv, phase = is_training)

                    activated_conv = lrelu(normalized_layer) #Right after conv layer, relu function has not yet specified.

                    prev = activated_conv