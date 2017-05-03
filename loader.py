import math, os
from PIL import Image
from glob import glob
import tensorflow as tf
import numpy as np
import librosa

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

        queue = tf.image.resize_area(queue, [self.scale_size[0], self.scale_size[1]])

        if self.data_format == 'NCHW':
            queue = tf.transpose(queue, [0, 3, 1, 2])
        elif self.data_format == 'NHWC':
            pass
        else:
            raise Exception("[!] Unkown data_format: {}".format(self.data_format))

        self.queue = tf.to_float(queue) / 255.0

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

            grid[h:h+h_width, w:w+w_width] = tensor[k] * 255
            k = k + 1
    return grid


class Spectrogram_Loader(object):
    def __init__(self, root, batch_size, scale_size, data_format, \
        split=None, seed=None, \
        sampling_rate = 16000, fft_size = 1024, offset = 2, duration = 3):

        self.root = root
        self.batch_size = batch_size
        self.scale_size = scale_size
        self.split = split
        self.data_format = data_format
        self.seed = seed
        self.sampling_rate = sampling_rate
        self.fft_size = fft_size
        self.offset = offset
        self.duration = duration # how long amount of audio slice do I want
        self.build_loader()
    
    def build_loader(self):

        dataset_name = os.path.basename(self.root)
        paths = glob("{}/".format(self.root))

        h = (self.fft_size/2)+1
        w = int(math.ceil(float(self.sampling_rate*self.duration)/(int(self.fft_size/2))))
        c = 1
        
        shape = [h,w,c]
        
        
        record_bytes = h*w*c

        filename_queue = tf.train.string_input_producer(list(paths), shuffle=False, seed=self.seed)
        # reader = tf.TextLineReader()
        reader = tf.FixedLengthRecordReader(record_bytes = record_bytes)
        # reader = tf.WholeFileReader()
        # reader = tf.TFRecordReader()
        filename, data = reader.read(filename_queue)
        print(filename)
        print(data)
        spectrogram = tf.decode_raw(data, tf.float32)
        # spectrogram = tf.decode_csv(data)
        spectrogram = tf.reshape(spectrogram, shape)
        # spectrogram.set_shape(shape)
        min_after_dequeue = 5000
        capacity = min_after_dequeue + 3*self.batch_size

        queue = tf.train.shuffle_batch(
            [spectrogram], batch_size=self.batch_size,
            num_threads=4, capacity=capacity,
            min_after_dequeue=min_after_dequeue, name='synthetic_inputs') 

        queue = tf.image.resize_area(queue, [self.scale_size[0], self.scale_size[1]])

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
    
    def make_spectrogram():

        #FIX ME : write your data directory
        domain_A_path = "/Users/Adrian/Desktop/Domain_A"
        domain_B_path = "/Users/Adrian/Desktop/Domain_B"
        domain_A_file_list = []
        domain_B_file_list = []


        # SAMPLING_RATE = 16000
        # FFT_SIZE = 1024 #Frequency resolution

        for (dirpath, dirnames, filenames) in walk(domain_A_path):
            domain_A_file_list.extend(filenames)
            break

        for (dirpath, dirnames, filenames) in walk(domain_B_path):
            domain_B_file_list.extend(filenames)
            break
            
        print("domain_A_file_list : ", domain_A_file_list, "\n", "domain_B_file_list : ", domain_B_file_list)



        Spectrogram_A_save = []
        Spectrogram_B_save = []

        #domain_A_file fetch from directory
        for audio in domain_A_file_list :

            #loading from offset(5s) to duration(8s)
            y, sr = librosa.core.load("/Users/Adrian/Desktop/Domain_A/" + audio, sr = self.sampling_rate, mono=True, offset=self.offset, duration=self.duration)
            D = librosa.core.stft(y=y, n_fft=self.fft_size, hop_length=int(self.fft_size/2), win_length=None, window='hann', center=True) # win_length = FFT_SIZE
            D = np.abs(D) # Magnitude of plain spectrogram
            # D = librosa.feature.melspectrogram(y=scope_y, n_fft=2048, hop_length=1024, sr=sr, n_mels=128, fmax=None) # use when you want to use mel-spectrogram
            
            Spectrogram_A_save.append(D)
            

        #domain_B_file fetch from directory
        for audio in domain_B_file_list :

            #loading from offset(5s) to duration(8s)
            y, sr = librosa.core.load("/Users/Adrian/Desktop/Domain_B/" + audio, sr = self.sampling_rate, mono=True, offset=self.offset, duration=self.duration)         
            D = librosa.core.stft(y = y, n_fft=self.fft_size, hop_length=int(self.fft_size/2), win_length=None, window='hann', center=True)
            D = np.abs(D) # Magnitude of plain spectrogram
            # D = librosa.feature.melspectrogram(y=scope_y, n_fft=2048, hop_length=1024, sr=sr, n_mels=128, fmax=None)#mel-spectrogram

            Spectrogram_B_save.append(D)
            
        print("Spectrogram A Shape : ", np.shape(Spectrogram_A_save[0]))
        print("Spectrogram B Shape : ", np.shape(Spectrogram_B_save[0]))
        print("Number of Spectrograms in Domain_A : ", len(Spectrogram_A_save))
        print("Number of Spectrograms in Domain_B : ", len(Spectrogram_B_save))

        Spectrogram_A_save = np.array(Spectrogram_A_save) #(Numberofspectrograms,col,row)
        Spectrogram_B_save = np.array(Spectrogram_B_save)
        print(Spectrogram_A_save.shape)
        print(Spectrogram_B_save.shape)

        try:
            for i, A in enumerate(Spectrogram_A_save):
                A.tofile("./spectrogram_files_A/spectrograms_A_{}.bin".format(i))
            for i, B in enumerate(Spectrogram_B_save):
                B.tofile("./spectrogram_files_B/spectrograms_B_{}.bin".format(i))
        except IOError :
            mkdir("./spectrogram_files_A")
            mkdir("./spectrogram_files_B")
            for i, A in enumerate(Spectrogram_A_save):
                A.tofile("./spectrogram_files_A/spectrograms_A_{}.bin".format(i))
            for i, B in enumerate(Spectrogram_B_save):
                B.tofile("./spectrogram_files_B/spectrograms_B_{}.bin".format(i))   


        plt.figure(figsize=(10, 4))
        #librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=None, x_axis='time')
        plt.title('Spectrogram')
        plt.imshow(np.log10(Spectrogram_A_save[0]+0.1), aspect = 'auto')
        plt.show()     