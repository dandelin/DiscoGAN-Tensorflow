import os
import math
import random
import tensorflow as tf
from loader import Loader, save_image


from custom_activation import lrelu



'''
gen_conv_infos = { "conv_layer_number": 4, "filter":[[4,4,3,64],[4,4,64,64*2],[4,4,64*2,64*4],[4,4,64*2,64*8], [4,4,64*2,1]],
                   "stride" : [[1,2,2,1],[1,2,2,1],[1,2,2,1],[1,2,2,1]] }

64 x 64 x 3   ->   31 x 31 x (64)  ->  ...


# Needs to be modified according to the structrue we are aiming for
'''





class DiscoGAN(object):
    def __init__(
        self, sess, a_dim=(64, 64), b_dim=(64, 64), channel=3, batch_size=64,
        gen_conv_infos, gen_deconv_infos, disc_conv_infos, phase = True      # Phase default : True (Training) 
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
        self.gen_conv_infos = gen_conv_infos
        self.gen_deconv_infos = gen_deconv_infos
        self.disc_conv_infos = disc_conv_infos
        self.loaders = {
            'A': Loader("imagesA", batch_size, a_dim, "NHWC"),
            'B':Loader("imagesB", batch_size, b_dim, "NHWC")
        }

    def build_generator(self, signature):

        # Build Generator Class
        in_dim = self.image_dims[signature[0]]
        out_dim = self.image_dims[signature[1]]
        
        
        g = Generator(in_dim=in_dim, out_dim=out_dim, channel=self.channel, batch_size=self.batch_size,
        conv_infos=self.gen_conv_infos, deconv_infos=self.gen_deconv_infos, signature=signature)

        setattr(self, 'G_' + signature, g)




    def build_discriminator(self, set_name):

        # Build Discriminator Class
        dim = self.image_dims[signature[set_name]]
        d = Discriminator(dim=dim, channel=self.channel, batch_size=self.batch_size,
        conv_infos=self.disc_conv_infos, set_name=set_name)
        setattr(self, 'D_' + set_name, d)


    def build_model(self, images_A, images_B):
        with tf.variable_scope('Fetch'):
            self.images_A = tf.placeholder(dtype=tf.float32,
                                     shape=[self.batch_size, 64, 64, 3],    #<<<-----------image shape-------------------
                                     name='images_A')
            self.images_B = tf.placeholder(dtype=tf.float32,
                                     shape=[self.batch_size, 64, 64, 3],    #<<<-----------image shape-------------------
                                     name='images_B')

        # Model Architecture
  
        self.build_generator(signature = 'AB') # Init G_AB
        self.build_generator(signature = 'BA') # Init G_BA
        self.build_discriminator(signature = 'A') # Init D_A
        self.build_discriminator(signature = 'B') # Init D_B

        # Domain_A -> Domain_B   &&   Domain_B -> Domain_A
        self.x_AB = G_AB.build_model(self.images_A) # Put x_A and generate x_AB
        self.x_BA = G_BA.build_model(self.images_B) # Put x_B and generate x_BA

        # Resconstruct 
        self.x_ABA = G_BA.build_model(self.x_AB) # Put x_AB and generate x_ABA
        self.x_BAB = G_BA.build_model(self.x_BA) # Put x_AB and generate x_ABA


        # Discriminate real images
        self.logits_real_A = D_A.build_model(self.x_A)  # Discriminate x_A
        self.logits_real_B = D_B.build_model(self.x_B)  # Discriminate x_B

        # Discriminate generated imaages
        self.logits_fake_A = D_A.build_model(self.x_BA)  # Discriminate x_BA
        self.logits_fake_B = D_B.build_model(self.x_AB)  # Discriminate x_AB




        ####### Loss #######

        #Discriminator Loss
        self.Discriminator_loss_A = -tf.log(self.logits_real_A) -tf.log(1-self.logits_fake_A) #L_D_A
        self.Discriminator_loss_B = -tf.log(self.logits_real_B) -tf.log(1-self.logits_fake_B) #L_D_B
        

        #Generator Loss
        self.Generator_loss_A = -tf.log(self.logits_fake_A) #L_GAN_A : Loss of generator(G_BA) trying to decive discriminator(D_B)
        self.Generator_loss_B = -tf.log(self.logits_fake_B) #L_GAN_B : Loss of generator(G_AB) trying to decive discriminator(D_A)

        #Reconstruction Loss : Three candidates according to the paper -> L1_norm, L2_norm, Huber Loss
        self.Reconstruction_loss_A = tf.nn.l2_loss(tf.subtract(self.x_ABA, x_A, name="Reconstruct_Error")) #L_CONST_A
        self.Reconstruction_loss_B = tf.nn.l2_loss(tf.subtract(self.x_BAB, x_B, name="Reconstruct_Error")) #L_CONST_B
          #for L1_norm : tf.losses.absolute_differences(labels, predictions)

        #Total Loss
        self.Discriminator_loss = self.Discriminator_loss_A + self.Discriminator_loss_B #L_D = L_D_A + L_D_B
        self.Generator_loss = (self.Generator_loss_B + self.Reconstruction_loss_A) + \
                        (self.Generator_loss_A + self.Reconstruction_loss_B)  #L_G = L_G_AB + L_G_BA = (L_GAN_B + L_CONST_A) + (L_GAN_A + L_CONST_B)


        return self.Generator_loss, self.Discriminator_loss

    def train(self, learning_rate = 0.002,  beta1 = 0.5, beta2 = 0.999, epsilon = 1e-08, epoch = 10000000):
        """ TO DO """
        self.lr = learning_rate
        self.B1 = beta1
        self.B2 = beta2
        self.eps = epsilon
        self.sess = tf.Session()
        self.epoch = epoch

"""     
        To Do
        
        Fetch Data using queue and feed it to self.images

        Pseudo code as follows :

        images = data.py()

        self.build_model(images)


        img_A = data_loader()
        img_B = data_loader()
        self.build_model(self, images_A, images_B)

"""

        trainable_variables = tf.trainable_variables() # get all variables that were checked "trainable = True"

        self.Generator_variables = [var for var in trainable_variables if 'G_' in var.name]        
        self.Discriminator_variables = [var for var in trainable_variables if 'D_' in var.name]


        for var in self.Generator_variables:
            print(var.name)
        for var in self.Discriminator_variables:
            print(var.name)


        # Optimizer for Generator and Descriminator each
        optimizer_G = tf.train.AdamOptimizer(learning_rate = self.lr, beta1=self.B1, beta2 = self.B2, epsilon = self.eps )
        optimizer_D = tf.train_AdamOptimizer(learning_rate = self.lr, beta1=self.B1, beta2 = self.B2, epsilon = self.eps )



        global_step = tf.Variable(0, name='global_step', trainable=False) #minibatch number

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #To force update moving average and variance

        with tf.control_dependencies(update_ops): # Ensures that we execute the update_ops before performing the train_step
            optimize_G = optimizer_G.minimize(self.Generator_loss, global_step = global_step, var_list = self.Generator_variables)
            optimize_D = optimizer_D.minimize(self.Discriminator_loss, global_step = global_step, var_list = self.Discriminator_variables)


        saver = tf.train.Saver(max_to_keep=1000)

        with self.sess() as sess:
            sess.run(tf.global_variables_initializer()) #run init
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord) # queue runners

            summary = tf.summary.merge_all() #merege summaries

            writer = tf.summary.FileWriter('./logs', sess.graph) # add the graph to the file './logs'
 
            """
            --------To Do----------

                Get Data Set

            images_A = data.fetch(path_A)    
            images_B = data.fetch(path_B)

            """

            
            for step in range(self.epoch):

            """ Pseudo code
                images_A = get_next_batch(images_A)
                images_B = get_next_batch(images_B)
            """    

                Generator_loss, Discriminator_loss = self.buildmodel(images_A, images_B)
                sess.run([optimize_G, optimize_D, self.Generator_loss, self.Discriminator_loss], \
                          feed_dict = {self.images_A : images_A, self.images_B : images_B} )

                if step % 100 == 0:
                    summary_run = sess.run(summary, feed_dict = {self.images_A : images_A, self.images_B : images_B})
                    writer.add_summary(summary, step)
                    
                if step % 10 == 0:
                    checkpoint_path = "/home/choi/Documents/git/DiscoGAN-Tensorflow/Checkpoint"
                    saver.save(sess, checkpoint_path, global_step = step)


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
        self.b_height = out_dim[0]
        self.b_width = out_dim[1]
        
        self.channel = channel
        self.signature = signature

        self.conv_infos = conv_infos
        self.deconv_infos = deconv_infos
    

    def build_model(self, image, reuse=False):
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

                normalized_layer = bn(conv)      # arg "phase" has to be specified whether it is training or test session         

                activated_conv = lrelu(normalized_layer) #Right after conv layer, relu function has not yet specified.

                prev = activated_conv




            for i, dconv_info in enumerate(deconv_infos):
                """
                    filter = 4 x 4, output_shape, strides = [1, stride, stride, 1]
                    add batch normalization
                    add tensorboard summaries
                """
                conv_t = tf.nn.conv2d_transpose(prev, filter, output_shape, strides,
                    padding="SAME", name="g_deconv_" + str(i))

                bn = batch_norm(name='g_bn_' + str(i))
                setattr(self, "bn_" + str(i), bn) 

                normalized_layer = bn(conv)              # arg "phase" has to be specified whether it is training or test session  

                activated_conv = lrelu(normalized_layer) # Right after conv layer, relu function has not yet specified.


                prev = conv_t


        Generated = prev 

        return Generated


class Discriminator(object):
    def __init__(
        self, dim=(64, 64), channel=3, batch_size=64, conv_infos, signature=None
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
                conv = tf.nn.conv2d(prev, conv_infos.filter[i], conv_infos.strides[i],
                    padding="SAME", name="g_conv_" + str(i))
                setattr(self, "conv_" + str(i), conv)

                if i == self.conv_layer_number-1 :
                    

                    Discriminated = tf.sigmoid(conv)  # no batch_norm and leaky relu activation at the last layer


                else :
                    
                    bn = batch_norm(name='g_bn_' + str(i))
                    setattr(self, "bn_" + str(i), bn) 

                    normalized_layer = bn(conv)      # arg "phase" has to be specified whether it is training or test session         

                    activated_conv = lrelu(normalized_layer) #Right after conv layer, relu function has not yet specified.

                    prev = activated_conv


        return Discriminated





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
        self.stride = conv_stride["stride"]
        self.current = 0
 
    
    def __iter__(self):
        return self
    
    def next(self):
        
        if self.current >= self.conv_layer_number:
            raise StopIteration
        else:
            self.current += 1
            return self

