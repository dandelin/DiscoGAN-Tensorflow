import os
from PIL import Image
from glob import glob
from loader import Spectrogram_Loader, Loader, save_image

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



# root = "./imageB"
# batch_size = 8
# scale_size = [64,64]
# data_format = "NHWC"

# loader = Loader(root, batch_size, scale_size, data_format, file_type="png")

# init_op = tf.global_variables_initializer()

# with tf.Session() as sess:

#     sess.run([init_op])
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess, coord=coord)

#     image = loader.get_image_from_loader(sess)
    
#     save_image(image, '{}/image.png'.format("test"))
    
#     coord.request_stop()
#     coord.join(threads)


root = "./log_spectrograms_male"
batch_size = 3
scale_size = [64,64]
data_format = "NHWC"
fft_size = 1024
sampling_rate = 16000
offset = 0
duration = 2.5

loader = Spectrogram_Loader(root, batch_size, scale_size, data_format, \
							offset = offset, fft_size = fft_size, sampling_rate = sampling_rate, duration = duration )

init_op = tf.global_variables_initializer()




with tf.Session() as sess:

    sess.run([init_op])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coord)
    
    spectrogram = loader.get_spectrogram_from_loader(sess)

    print(spectrogram.shape)
    spectrogram = spectrogram[0,:,:,0]
    print(spectrogram.shape)
    print(spectrogram)



    


    loader.save_reconstructed_audio(spectrogram, "test.wav")

    spec_plot = plt.imshow(np.log10(spectrogram+0.1), aspect = 'auto')
    plt.title('Spectrogram')
    plt.ylabel('Frequency Bin Index')
    plt.xlabel('Time Frame Index')
    plt.show()

    coord.request_stop()
    coord.join(threads)