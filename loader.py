import math, os
from PIL import Image
from glob import glob
import tensorflow as tf
import numpy as np

class Loader(object):
    def __init__(self, root, batch_size, scale_size, data_format, \
        file_type="png", split=None, is_grayscale=False, seed=None):

        self.root = root
        self.batch_size = batch_size
        self.scale_size = scale_size
        self.data_format = data_format
        self.split = split
        self.is_grayscale = is_grayscale
        self.seed = seed
        
        self.build_loader(file_type)
    
    def build_loader(self, file_type):
        dataset_name = os.path.basename(self.root)
        paths = glob("{}/*.{}".format(self.root, file_type))

        if file_type == "jpg":
            tf_decode = tf.image.decode_jpeg
        elif file_type == "png":
            tf_decode = tf.image.decode_png

        img = Image.open(paths[0])
        w, h = img.size
        shape = [h, w, 3]

        filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=self.seed)
        reader = tf.WholeFileReader()
        filename, data = reader.read(filename_queue)
        image = tf_decode(data, channels=3)

        if self.is_grayscale:
            image = tf.image.rgb_to_grayscale(image)
        image.set_shape(shape)

        min_after_dequeue = 5000
        capacity = min_after_dequeue + 3 * self.batch_size

        queue = tf.train.shuffle_batch(
            [image], batch_size=self.batch_size,
            num_threads=4, capacity=capacity,
            min_after_dequeue=min_after_dequeue, name='synthetic_inputs') 

        queue = tf.image.resize_nearest_neighbor(queue, [self.scale_size[0], self.scale_size[1]])

        if self.data_format == 'NCHW':
            queue = tf.transpose(queue, [0, 3, 1, 2])
        elif self.data_format == 'NHWC':
            pass
        else:
            raise Exception("[!] Unkown data_format: {}".format(self.data_format))

        self.queue = tf.to_float(queue)

    def get_image_from_loader(self, sess):
        x = self.queue.eval(session=sess)
        if self.data_format == 'NCHW':
            x = x.transpose([0, 2, 3, 1])
        return x

def save_image(tensor, filename, nrow=8, padding=2,
            normalize=False, scale_each=False):
    ndarr = make_grid(tensor, nrow=nrow, padding=padding,
                            normalize=normalize, scale_each=scale_each)
    im = Image.fromarray(ndarr)
    im.save(filename)

def make_grid(tensor, nrow=8, padding=2,
            normalize=False, scale_each=False):
    """Code based on https://github.com/pytorch/vision/blob/master/torchvision/utils.py"""
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[1] + padding), int(tensor.shape[2] + padding)
    grid = np.zeros([height * ymaps + 1 + padding // 2, width * xmaps + 1 + padding // 2, 3], dtype=np.uint8)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            h, h_width = y * height + 1 + padding // 2, height - padding
            w, w_width = x * width + 1 + padding // 2, width - padding

            grid[h:h+h_width, w:w+w_width] = tensor[k]
            k = k + 1
    return grid
        