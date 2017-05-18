import os
import math
import random
import tensorflow as tf

from loader import Loader, save_image, Spectrogram_Loader
from generator import Generator
from discriminator import Discriminator
from util import ReconstructionLoss, GANLoss


class DiscoGAN(object):
    def __init__(
        self, sess, gen_conv_infos, gen_deconv_infos, disc_conv_infos,
        a_dim=[64, 64], b_dim=[64, 64], channel=3, batch_size=64, config=None
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

        self.loader_a = Spectrogram_Loader('./Test_spectrograms_female', self.batch_size, self.image_dims['A'], "NHWC")
        self.loader_b = Spectrogram_Loader('./Test_spectrograms_male', self.batch_size, self.image_dims['B'], "NHWC")
        
        self.channel = channel
        self.gen_conv_infos = gen_conv_infos
        self.gen_deconv_infos = gen_deconv_infos
        self.disc_conv_infos = disc_conv_infos
        self.config = config

    def build_generator(self, signature):
        g = Generator(conv_infos=self.gen_conv_infos, deconv_infos=self.gen_deconv_infos, signature=signature)
        setattr(self, 'G_' + signature, g)
        return g


    def build_discriminator(self, signature):
        d = Discriminator(conv_infos=self.disc_conv_infos, signature=signature)
        setattr(self, 'D_' + signature, d)
        return d


    def build_model(self):
        with tf.variable_scope('is_training'):
            is_training = tf.get_variable('is_training', dtype=tf.bool, initializer=True)

        # self.x_A = Loader("./imageA", self.batch_size, self.image_dims['A'], "NHWC", file_type="jpg").queue
        # self.x_B = Loader("./imageB", self.batch_size, self.image_dims['B'], "NHWC", file_type="jpg").queue

        self.x_A = self.loader_a.queue
        self.x_B = self.loader_b.queue

        # Model Architecture
        self.build_generator(signature = 'AB') # Init G_AB
        self.build_generator(signature = 'BA') # Init G_BA
        self.build_discriminator(signature = 'A') # Init D_A
        self.build_discriminator(signature = 'B') # Init D_B

        # Domain_A -> Domain_B   &&   Domain_B -> Domain_A
        self.x_AB = self.G_AB.build_model(self.x_A) # Put x_A and generate x_AB
        self.x_BA = self.G_BA.build_model(self.x_B) # Put x_B and generate x_BA

        # Resconstruct 
        self.x_ABA = self.G_BA.build_model(self.x_AB, reuse=True) # Put x_AB and generate x_ABA
        self.x_BAB = self.G_AB.build_model(self.x_BA, reuse=True) # Put x_AB and generate x_ABA


        # Discriminate real images
        self.logits_real_A = self.D_A.build_model(self.x_A)  # Discriminate x_A
        self.logits_real_B = self.D_B.build_model(self.x_B)  # Discriminate x_B

        # Discriminate generated imaages
        self.logits_fake_BA = self.D_A.build_model(self.x_BA, reuse=True)  # Discriminate x_BA
        self.logits_fake_AB = self.D_B.build_model(self.x_AB, reuse=True)  # Discriminate x_AB

        ####### Loss #######

        # Real / Fake GAN loss A
        self.loss_real_A = GANLoss(logits=self.logits_real_A, is_real=True)
        self.loss_fake_A = GANLoss(logits=self.logits_fake_BA, is_real=False)
        
        # Real / Fake GAN loss B
        self.loss_real_B = GANLoss(logits=self.logits_real_B, is_real=True)
        self.loss_fake_B = GANLoss(logits=self.logits_fake_AB, is_real=False)

        # Losses of Discriminator
        self.loss_Discriminator_A = self.loss_real_A + self.loss_fake_A # L_D_A in paper notation
        self.loss_Discriminator_B = self.loss_real_B + self.loss_fake_B # L_D_B in paper notation

        # Generator GAN Loss
        self.loss_GAN_AB = GANLoss(logits=self.logits_fake_AB, is_real=True)
        self.loss_GAN_BA = GANLoss(logits=self.logits_fake_BA, is_real=True)

        #Reconstruction Loss : Three candidates according to the paper -> L1_norm, L2_norm, Huber Loss
        self.Reconstruction_loss_A = ReconstructionLoss(self.x_A, self.x_ABA) #L_CONST_A
        self.Reconstruction_loss_B = ReconstructionLoss(self.x_B, self.x_BAB) #L_CONST_B

        # Generator Loss
        self.loss_Generator_AB = self.loss_GAN_AB + self.Reconstruction_loss_A
        self.loss_Generator_BA = self.loss_GAN_BA + self.Reconstruction_loss_B

        #Total Loss
        self.loss_Discriminator = self.loss_Discriminator_A + self.loss_Discriminator_B #L_D = L_D_A + L_D_B
        self.loss_Generator = self.loss_Generator_AB + self.loss_Generator_BA  #L_G = L_G_AB + L_G_BA = (L_GAN_B + L_CONST_A) + (L_GAN_A + L_CONST_B)

        trainable_variables = tf.trainable_variables() # get all variables that were checked "trainable = True"
        self.G_vars = [var for var in trainable_variables if 'G_' in var.name]        
        self.D_vars = [var for var in trainable_variables if 'D_' in var.name]
        
        # Add summaries.
        # Add loss summaries
        tf.summary.scalar("losses/loss_Discriminator", self.loss_Discriminator)
        tf.summary.scalar("losses/loss_Discriminator_A", self.loss_Discriminator_A)
        tf.summary.scalar("losses/loss_Discriminator_B", self.loss_Discriminator_B)
        tf.summary.scalar("losses/loss_Generator", self.loss_Generator)
        tf.summary.scalar("losses/loss_Generator_AB", self.loss_Generator_AB)
        tf.summary.scalar("losses/loss_Generator_BA", self.loss_Generator_BA)

        # Add histogram summaries
        for var in self.D_vars:
            tf.summary.histogram(var.op.name, var)
        for var in self.G_vars:
            tf.summary.histogram(var.op.name, var)

        # Add image summaries
        tf.summary.image('x_A', self.x_A, max_outputs=4)
        tf.summary.image('x_B', self.x_B, max_outputs=4)
        tf.summary.image('x_AB', self.x_AB, max_outputs=4)
        tf.summary.image('x_BA', self.x_BA, max_outputs=4)
        tf.summary.image('x_ABA', self.x_ABA, max_outputs=4)
        tf.summary.image('x_BAB', self.x_BAB, max_outputs=4)

    def train(self, learning_rate = 0.002, beta1 = 0.5, beta2 = 0.999, epsilon = 1e-08, iteration = 10000000):
        """ TO DO """
        self.lr = learning_rate
        self.B1 = beta1
        self.B2 = beta2
        self.eps = epsilon
        self.sess = tf.Session()
        self.iteration = iteration

        # Optimizer for Generator and Descriminator each
        optimizer_G = tf.train.AdamOptimizer(learning_rate = self.lr, beta1=self.B1, beta2 = self.B2, epsilon = self.eps )
        optimizer_D = tf.train.AdamOptimizer(learning_rate = self.lr, beta1=self.B1, beta2 = self.B2, epsilon = self.eps )

        global_step = tf.Variable(0, name='global_step', trainable=False) #minibatch number

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #To force update moving average and variance

        with tf.control_dependencies(update_ops): # Ensures that we execute the update_ops before performing the train_step
            optimize_G = optimizer_G.minimize(self.loss_Generator, global_step = global_step, var_list = self.G_vars)
            optimize_D = optimizer_D.minimize(self.loss_Discriminator, global_step = global_step, var_list = self.D_vars)

        saver = tf.train.Saver(max_to_keep=1000)

        summary = tf.summary.merge_all() #merege summaries

        os.makedirs(self.config.log_dir, exist_ok=True)
        writer = tf.summary.FileWriter('./{}'.format(self.config.log_dir), self.sess.graph) # add the graph to the file './logs'

        with tf.variable_scope('is_training', reuse=True):
            is_training = tf.get_variable('is_training', dtype=tf.bool)
            self.sess.run(tf.assign(is_training, True))

        self.sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()

        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        
        for step in range(self.iteration):
            if coord.should_stop():
                break

            _, _, summary_run = self.sess.run([optimize_G, optimize_D, summary])
            
            if step % 100 == 0:
                writer.add_summary(summary_run, step)
                os.makedirs(self.config.snapshot_dir, exist_ok=True)
                for t in ['A', 'AB', 'BA', 'B', 'ABA', 'BAB']:
                    arg = getattr(self, 'x_{}'.format(t))
                    images = self.sess.run(arg)
                    save_image(images, '{}/{}{}.png'.format(self.config.snapshot_dir, t, step))
                    
                    
            if step % 100 == 0:
                writer.add_summary(summary_run, step)
                os.makedirs(self.config.audio_dir, exist_ok=True)
                for t in ['A', 'AB', 'BA', 'B', 'ABA', 'BAB']:
                    arg = getattr(self, 'x_{}'.format(t))
                    spectrograms = self.sess.run(arg)
                    print("spectrogram first sample", '{}{}'.format(t,step), "shape : ", spectrograms[0,:,:,0].shape)
                    spectrograms = self.sess.run(arg)            
                    print("reconstruction session...")
                    self.loader_a.save_reconstructed_audio(spectrograms[0,:,:,0], '{}/{}{}.wav'.format(self.config.audio_dir, t, step))
                    

            if step % 500 == 0:
                os.makedirs(self.config.checkpoint_dir, exist_ok=True)
                saver.save(self.sess, "{}/model.ckpt".format(self.config.checkpoint_dir), global_step = step)
        coord.request_stop()
        coord.join(threads)