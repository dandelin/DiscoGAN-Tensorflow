import tensorflow as tf
from util import lrelu, conv_layer, conv_layer_t, batch_norm

class Discriminator(object):
    def __init__(self, conv_infos, dim=(64, 64), channel=3, signature=None):
        self.dim = dim
        self.channel = channel
        self.signature = signature
        self.conv_infos = conv_infos

    def build_model(self, image, reuse=False):
        with tf.variable_scope('is_training', reuse=True):
            is_training = tf.get_variable('is_training', dtype=tf.bool)

        with tf.variable_scope("D_" + self.signature) as scope:
            if reuse:
                scope.reuse_variables()

            conv_num = self.conv_infos['conv_layer_number']
            conv_filter = self.conv_infos['filter']
            conv_stride = self.conv_infos['stride']

            prev = image

            for i in range(conv_num):
                if i == 0 or i == conv_num - 1:
                    prev = conv_layer(prev, conv_filter[i], "d_conv_{}".format(i), activation=lrelu, batch_norm=None, reuse=reuse)
                else:
                    bn = batch_norm(name="d_bn_{}".format(i))
                    prev = conv_layer(prev, conv_filter[i], "d_conv_{}".format(i), activation=lrelu, batch_norm=bn, reuse=reuse)

                setattr(self, "conv_{}".format(i), prev)
            return tf.sigmoid(prev)