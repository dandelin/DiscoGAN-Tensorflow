from model import DiscoGAN
from loader import Spectrogram_Loader, Loader, save_image
import tensorflow as tf

batch_size = 128

gen_conv_infos = {
    "conv_layer_number": 4,
    "filter":[
        [4,4,3,64],
        [4,4,64,64*2],
        [4,4,64*2,64*4],
        [4,4,64*4,64*8],
    ],
    "stride" : [
        [1,2,2,1],
        [1,2,2,1],
        [1,2,2,1],
        [1,2,2,1],
    ],
}

gen_deconv_infos = {
    "conv_layer_number": 4,
    "filter":[
        [4,4,64*4,64*8],
        [4,4,64*2,64*4],
        [4,4,64*1,64*2],
        [4,4,3,64],
    ],
    "stride" : [
        [1,2,2,1],
        [1,2,2,1],
        [1,2,2,1],
        [1,2,2,1],
    ],
    "output_dims" : [
        [batch_size, 8, 8, 64*4],
        [batch_size, 16, 16, 64*2],
        [batch_size, 32, 32, 64*1],
        [batch_size, 64, 64, 3]
    ],
}

disc_conv_infos = {
    "conv_layer_number": 5,
    "filter":[
        [4,4,3,64],
        [4,4,64,64*2],
        [4,4,64*2,64*4],
        [4,4,64*4,64*8],
        [4,4,64*8,1],
    ],
    "stride" : [
        [1,2,2,1],
        [1,2,2,1],
        [1,2,2,1],
        [1,2,2,1],
        [1,1,1,1],
    ],
}

if __name__ == "__main__":

    with tf.Session() as sess:

        discoGAN = DiscoGAN(sess, gen_conv_infos, gen_deconv_infos, disc_conv_infos, batch_size=batch_size)

        discoGAN.build_model()
        discoGAN.train()