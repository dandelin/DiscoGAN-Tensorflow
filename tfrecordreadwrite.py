import numpy as numpy
import tensorflow as tf

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# spectrograms and labels array as input
def convert_to(spectrograms, name):
  num_examples = spectrograms.shape[0]
  rows = spectrograms.shape[1]
  cols = spectrograms.shape[2]
  depth = spectrograms.shape[3] #channel

  print(num_examples, rows, cols, depth)

  filename = name
  print('Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  for index in range(num_examples):
    spectrograms_raw = spectrograms[index].tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'depth': _int64_feature(depth),
        'spectrograms_raw': _bytes_feature(spectrograms_raw)}))
    writer.write(example.SerializeToString())

    # Remember to generate a file name queue of you 'train.TFRecord' file path
def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example, \
    features = {'spectrograms_raw': tf.FixedLenFeature([], tf.string)})
  spectrograms = tf.decode_raw(features['spectrograms_raw'], tf.float32)




  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  # image = tf.cast(image, tf.float32)
  # image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  

  return spectrograms