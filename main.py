from model import DiscoGAN
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
        [4,4,64*8,64*4],
        [4,4,64*4,64*2],
        [4,4,64*2,64],
        [4,4,64,3],
    ],
    "stride" : [
        #[1,2,2,1],
        [1,2,2,1],
        [1,2,2,1],
        [1,2,2,1],
        [1,2,2,1],
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
        discoGAN.build_model()
        discoGAN.train()