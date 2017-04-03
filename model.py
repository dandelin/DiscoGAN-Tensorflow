import os
import math
import random
import numpy as np
import tensorflow as tf

from custom_activation import lrelu









class DiscoGAN(object):
    def __init__(
        self, sess, a_dim=(64, 64), b_dim=(64, 64), channel=3, batch_size=64,
        conv_infos, deconv_infos, learning_rate=0.0002 
    ):
        """
        Arguments :
            sess : Tensorflow Session
            x_size : (height, width) information of set X
            channel : 3 (RGB), 1 (Greyscale, Spectrogram)
            batch_size : minibatch size
            conv_infos : convolutional layer's informations
            deconv_infos : deconvolutional layer's informations
            signature : Instance's name like G_AB
        """
        self.sess = sess
        self.batch_size = batch_size
        
        self.image_dims = {
            'A': a_dim,
            'B': b_dim
        }
        
        self.channel = channel
        self.conv_infos = conv_infos
        self.deconv_infos = deconv_infos

        self.lr = learning_rate
    

        x_a = tf.placeholder(tf.float32,shape=[batch_size,a_dim,channel],name="x_a")
        x_b = tf.placeholder(tf.float32,shape=[batch_size,b_dim,channel],name="x_b")



    def build_generator(self, signature):
        in_dim = self.image_dims[signature[0]]
        out_dim = self.image_dims[signature[1]]
        g = Generator(in_dim=in_dim, out_dim=out_dim, channel=self.channel, batch_size=self.batch_size,
        conv_infos=self.conv_infos, deconv_infos=self.deconv_infos, signature=signature)
        setattr(self, 'G_' + signature, g)

    def build_discriminator(self, set_name):
        dim = self.image_dims[signature[set_name]]
        d = Discriminator(dim=dim, channel=self.channel, batch_size=self.batch_size,
        conv_infos=self.conv_infos, set_name=set_name)
        setattr(self, 'D_' + set_name, d)

    def train(self):
        """ TO DO """
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #To force update moving average and variance
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())


class Generator(object):
    def __init__(
        self, in_dim=(64, 64), out_dim=(64, 64), channel=3, batch_size=64,
        conv_infos, deconv_infos, bn_number, signature='AB'
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
        self.sess = sess
        self.batch_size = batch_size
        
        self.in_dim = in_dim
        self.a_height = in_dim[0]
        self.a_width = in_dim[1]
        
        self.out_dim = out_dim
        self.b_height = out_dim[9]
        self.b_width = out_dim[1]
        
        self.channel = channel
        self.signature = signature
        self.conv_infos = conv_infos
        self.deconv_infos = deconv_infos
    

    def build_model(self, image, reuse=False):
        with tf.variable_scope("G_" + this.signature) as scope:
            if reuse:
                scope.reuse_variables()

            for i in bn_number:

                bn = batch_norm(name='g_bn_' + str(i))
                setattr(self, "bn_" + str(i), bn)
 



            prev = image

            for i, conv_info in enumerate(conv_infos):
                """
                    filter = 4 x 4, strides = [1, stride, stride, 1]
                    add batch normalization
                    add tensorboard summaries
                """
                conv = tf.nn.conv2d(prev, filter, strides,
                    padding="SAME", name="g_conv_" + str(i))
                setattr(self, "conv_" + str(i), conv)
                prev = conv

            for i, dconv_info in enumerate(deconv_infos):
                """
                    filter = 4 x 4, output_shape, strides = [1, stride, stride, 1]
                    add batch normalization
                    add tensorboard summaries
                """
                conv_t = tf.nn.conv2d_transpose(prev, filter, output_shape, strides,
                    padding="SAME", name="g_deconv_" + str(i))
                prev = conv_t

class Discriminator(object):
    def __init__(
        self, dim=(64, 64), channel=3, batch_size=64, conv_infos, set_name='A'
    ):
        """
        Arguments :
            dim : (height, width) information
            channel : 3 (RGB), 1 (Greyscale, Spectrogram)
            batch_size : minibatch size
            conv_infos : convolutional layer's informations
            set_name : Instance's name like D_A
        """
        self.batch_size = batch_size
        
        self.dim = dim
        
        self.channel = channel
        self.set_name = set_name
        self.conv_infos = conv_infos
    
    def build_model(self, image, reuse=False):
        """ TO DO """


class batch_norm(object):
    def __init__(self, epsilon = 0.001, momentum = 0.99, name = "batch_norm"):
        self.epsilon = epsilon
        self.momentum = momentum
        self.name = name

    def __call__(self, x, phase = True):   # If phase == True, then training phase. If phase == False, then test phase.
        tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,  #The more you have your data, the bigger the momentum has to be
                      updates_collections=None, 
                      epsilon=self.epsilon,
                      scale=True,  # If next layer is linear like "Relu", this can be set as "False". Because You don't really need gamma in this case.
                      is_training=phase,  # Training mode or Test mode 
                      scope=self.name)