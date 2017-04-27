from model import DiscoGAN
from loader import Loader, save_image
import tensorflow as tf

gen_conv_infos = {
    "conv_layer_number": 4,
    "filter":[
        [4,4,3,64],
        [4,4,64,64*2],
        [4,4,64*2,64*4],
        [4,4,64*4,64*8],
        #[4,4,64*8,100],
    ],
    "stride" : [
        #[1,2,2,1],
        [1,2,2,1],
        [1,2,2,1],
        [1,2,2,1],
        [1,2,2,1],
    ],
    # "padding": [
    #     1, 1, 1, 1
    # ]
}

gen_deconv_infos = {
    "conv_layer_number": 4,
    "filter":[
        #[4,4,100,64*8],
        [4,4,64*4,64*8],
        [4,4,64*2,64*4],
        [4,4,64*1,64*2],
        [4,4,3,64],
    ],
    "stride" : [
        #[1,2,2,1],
        [1,2,2,1],
        [1,2,2,1],
        [1,2,2,1],
        [1,2,2,1],
    ],
    "output_dims" : [
        [64, 8, 8, 64*4],
        [64, 16, 16, 64*2],
        [64, 32, 32, 64*1],
        [64, 64, 64, 3]
    ],
    # "padding" : [
    #     1, 1, 1, 1
    # ]
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
    # "padding": [
    #     1, 1, 1, 1
    # ]
}

if __name__ == "__main__":

    with tf.Session() as sess:

        discoGAN = DiscoGAN(sess, gen_conv_infos, gen_deconv_infos, disc_conv_infos)

        sess.run(tf.global_variables_initializer()) #run init
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord) # queue runners

        discoGAN.build_model()
        discoGAN.train()

        coord.request_stop()
        coord.join(threads)