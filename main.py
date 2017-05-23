from model import DiscoGAN
from loader import Spectrogram_Loader, Loader, save_image
from config import config, adict
import tensorflow as tf

shape = adict()
shape.width = 255
shape.height = 513
shape.channel = 1
c = config('spectro', shape)

if __name__ == "__main__":

    with tf.Session() as sess:


        discoGAN = DiscoGAN(sess, c.gen_conv_infos, c.gen_deconv_infos, c.disc_conv_infos,
            batch_size=c.batch_size, config=c, a_dim=[shape.height, shape.width], b_dim=[shape.height, shape.width], channel=shape.channel)

        discoGAN.build_model()
        discoGAN.train()