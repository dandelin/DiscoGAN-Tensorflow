import tensorflow as tf
from util import lrelu, conv_information, batch_norm

class Generator(object):
    def __init__(
        self, conv_infos, deconv_infos, bn_number,
        in_dim=(64, 64), out_dim=(64, 64), channel=3, batch_size=64, signature='AB'
    ):
        """
        Arguments :
            x_dim : (height, width) information of set X
            channel : 3 (RGB), 1 (Greyscale, Spectrogram)
            batch_size : minibatch size
            conv_infos : convolutional layer's informations
            deconv_infos : deconvolutional layer's informations
            signature : Instance's name like G_AB
        """
        self.batch_size = batch_size
        
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
        with tf.variable_scope('is_training'):
            is_training = tf.get_variable('is_training')
        with tf.variable_scope("G_" + self.signature) as scope:
            if reuse:
                scope.reuse_variables()
                
            
            conv_infos = conv_information(self.conv_infos) #make instance


            prev = image

            for i, conv_info in enumerate(conv_infos):
                """
                    filter = 4 x 4, strides = [1, stride, stride, 1]
                    add batch normalization
                    add tensorboard summaries
                """
                conv = tf.nn.conv2d(prev, conv_info.filter[i], conv_info.strides[i],
                    padding="SAME", name="g_conv_" + str(i))
                setattr(self, "conv_" + str(i), conv)

                bn = batch_norm(name='g_bn_' + str(i))
                setattr(self, "bn_" + str(i), bn) 

                normalized_layer = bn(conv, phase=is_training)      # arg "phase" has to be specified whether it is training or test session         

                activated_conv = lrelu(normalized_layer) #Right after conv layer, relu function has not yet specified.

                prev = activated_conv

            
            deconv_infos = conv_information(self.deconv_infos)


            for i, deconv_info in enumerate(deconv_infos):
                """
                    filter = 4 x 4, output_shape, strides = [1, stride, stride, 1]
                    add batch normalization
                    add tensorboard summaries
                """
                conv_t = tf.nn.conv2d_transpose(prev, deconv_info.filter[i], self.out_dim, deconv_info.stride[i],
                    padding="SAME", name="g_deconv_" + str(i))
                
                if i == deconv_infos.conv_layer_number - 1:
                    return tf.sigmoid(conv_t)

                bn = batch_norm(name='g_bn_' + str(i))
                setattr(self, "bn_" + str(i), bn) 

                normalized_layer = bn(conv, phase=is_training)              # arg "phase" has to be specified whether it is training or test session  

                activated_conv = tf.nn.relu(normalized_layer) # Right after conv layer, relu function has not yet specified.

                prev = conv_t