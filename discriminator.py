import tensorflow as tf
from util import lrelu, conv_information, batch_norm

class Discriminator(object):
    def __init__(
        self, conv_infos, dim=(64, 64), channel=3, batch_size=64, signature=None
        ):
        """
        Arguments :
            dim : (height, width) information
            channel : 3 (RGB), 1 (Greyscale, Spectrogram)
            batch_size : minibatch size
            conv_infos : convolutional layer's informations
            signature : Instance's name like D_A
        """
        self.batch_size = batch_size
        
        self.dim = dim
        
        self.channel = channel
        self.signature = signature
        self.conv_infos = conv_infos
    
    def build_model(self, image, reuse=False):
        """ TO DO """
        with tf.variable_scope("D_" + self.signature) as scope:
            if reuse:
                scope.reuse_variables()
                
            
            conv_infos = conv_information(self.conv_infos) #make conv_infos instance


            prev = image

            for i, conv_info in enumerate(conv_infos):
                """
                    filter = 4 x 4, strides = [1, stride, stride, 1]
                    add batch normalization
                    add tensorboard summaries
                """
                conv = tf.nn.conv2d(prev, conv_infos.filter[i], conv_infos.stride[i],
                    padding="SAME", name="g_conv_" + str(i))
                setattr(self, "conv_" + str(i), conv)

                if i == conv_info.conv_layer_number-1 :
                    

                    Discriminated = tf.sigmoid(conv)  # no batch_norm and leaky relu activation at the last layer


                else :
                    
                    bn = batch_norm(name='g_bn_' + str(i))
                    setattr(self, "bn_" + str(i), bn) 

                    normalized_layer = bn(conv)      # arg "phase" has to be specified whether it is training or test session         

                    activated_conv = lrelu(normalized_layer) #Right after conv layer, relu function has not yet specified.

                    prev = activated_conv


        return Discriminated