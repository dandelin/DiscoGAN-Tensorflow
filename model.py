import os
import math
import random
import tensorflow as tf

from loader import Loader, save_image
from generator import Generator
from discriminator import Discriminator
from util import lrelu, conv_information, batch_norm


'''
gen_conv_infos = { "conv_layer_number": 4, "filter":[[4,4,3,64],[4,4,64,64*2],[4,4,64*2,64*4],[4,4,64*2,64*8], [4,4,64*2,1]],
                   "stride" : [[1,2,2,1],[1,2,2,1],[1,2,2,1],[1,2,2,1]] }

64 x 64 x 3   ->   31 x 31 x (64)  ->  ...


# Needs to be modified according to the structrue we are aiming for
'''





class DiscoGAN(object):
    def __init__(
        self, sess, gen_conv_infos, gen_deconv_infos, disc_conv_infos,
        a_dim=(64, 64), b_dim=(64, 64), channel=3, batch_size=64, phase = True      # Phase default : True (Training) 
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
            'A': Loader("imagesA", batch_size, a_dim, "NHWC"), #Number_batch, Height, Width, Channel
            'B': Loader("imagesB", batch_size, b_dim, "NHWC")
        }

    def build_generator(self, signature):

        # Build Generator Class
        in_dim = self.image_dims[signature[0]]
        out_dim = self.image_dims[signature[1]]
        
        
        g = Generator(in_dim=in_dim, out_dim=out_dim, channel=self.channel, batch_size=self.batch_size,
        conv_infos=self.gen_conv_infos, deconv_infos=self.gen_deconv_infos, signature=signature)

        setattr(self, 'G_' + signature, g)


    def build_discriminator(self, signature):

        # Build Discriminator Class
        dim = self.image_dims[signature]
        d = Discriminator(dim=dim, channel=self.channel, batch_size=self.batch_size,
        conv_infos=self.disc_conv_infos, signature=signature)
        setattr(self, 'D_' + signature, d)


    def build_model(self, images_A, images_B):
        with tf.variable_scope('is_training'):
            is_training = tf.get_variable('is_training', dtype=tf.bool)
        with tf.variable_scope('Fetch'):
            self.images_A = tf.placeholder(self.loaders['A'],
                                    dtype=tf.float32,
                                    shape=[self.batch_size, 64, 64, 3],    #<<<-----------image shape(NHWC)-------------------
                                    name='images_A')
            self.images_B = tf.placeholder(self.loaders['B'],
                                    dtype=tf.float32,
                                    shape=[self.batch_size, 64, 64, 3],    #<<<-----------image shape(NHWC)-------------------
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

    def train(self, learning_rate = 0.002,  beta1 = 0.5, beta2 = 0.999, epsilon = 1e-08, iteration = 10000000):
        """ TO DO """
        self.lr = learning_rate
        self.B1 = beta1
        self.B2 = beta2
        self.eps = epsilon
        self.sess = tf.Session()
        self.iteration = iteration

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
        optimizer_D = tf.train.AdamOptimizer(learning_rate = self.lr, beta1=self.B1, beta2 = self.B2, epsilon = self.eps )



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

            with tf.variable_scope('is_training'):
                is_training = tf.get_variable('is_training')
                sess.run(tf.assign(is_training, True))

            
            for step in range(self.iteration):

                """ Pseudo code
                    images_A = get_next_batch(images_A)
                    images_B = get_next_batch(images_B)
                """    

                Generator_loss, Discriminator_loss = self.build_model(images_A, images_B)
                sess.run([optimize_G, optimize_D])
                          


                summary_run = sess.run(summary)
                writer.add_summary(summary, step)
                    
                if step % 50 == 0:
                    checkpoint_path = "/home/choi/Documents/git/DiscoGAN-Tensorflow/Checkpoint"
                    saver.save(sess, checkpoint_path, global_step = step)