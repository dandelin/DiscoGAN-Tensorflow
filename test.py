import os
from PIL import Image
from glob import glob
import tensorflow as tf
from loader import Loader, save_image
import numpy as np


root = "./imageB"
batch_size = 8
scale_size = [64,64]
data_format = "NHWC"

loader = Loader(root, batch_size, scale_size, data_format, file_type="png")

init_op = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run([init_op])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coord)

    image = loader.get_image_from_loader(sess)
    
    save_image(image, '{}/image.png'.format("test"))
    
    coord.request_stop()
    coord.join(threads)