import tensorflow as tf
from util import lrelu, conv_layer, conv_layer_t, batch_norm

class Generator(object):
    def __init__(self, conv_infos, deconv_infos, signature='AB'):
        self.signature = signature
        self.conv_infos = conv_infos
        self.deconv_infos = deconv_infos

    def build_model(self, image, reuse=False):
        with tf.variable_scope('is_training', reuse=True):
            is_training = tf.get_variable('is_training', dtype=tf.bool)
        with tf.variable_scope("G_" + self.signature) as scope:
            if reuse:
                scope.reuse_variables()
            
            conv_num = self.conv_infos['conv_layer_number']
            deconv_num = self.deconv_infos['conv_layer_number']
            conv_filter = self.conv_infos['filter']
            deconv_filter = self.deconv_infos['filter']
            conv_stride = self.conv_infos['stride']
            deconv_stride = self.deconv_infos['stride']
            output_dims = self.deconv_infos['output_dims']

            prev = image
            
            for i in range(conv_num):
                if i == 0:
                    prev = conv_layer(prev, conv_filter[i], "g_conv_{}".format(i), activation=lrelu, batch_norm=None, reuse=reuse)
                else:
                    bn = batch_norm(name="g_cbn_{}".format(i))
                    prev = conv_layer(prev, conv_filter[i], "g_conv_{}".format(i), activation=lrelu, batch_norm=bn, reuse=reuse)

                setattr(self, "conv_{}".format(i), prev)

            for i in range(deconv_num):
                if i != self.deconv_infos['conv_layer_number'] - 1:
                    bn = batch_norm(name="g_dbn_{}".format(i))
                    prev = conv_layer_t(prev, deconv_filter[i], "g_deconv_{}".format(i), output_dims[i], batch_norm=bn, reuse=reuse)
                else:
                    prev = conv_layer_t(prev, deconv_filter[i], "g_deconv_{}".format(i), output_dims[i], batch_norm=None, reuse=reuse)
                
                setattr(self, "deconv_{}".format(i), prev)

            return tf.tanh(prev)