"""Checks if a set of TFRecords are valid

Author: @developmentseed
Run:
    python3 verify_tfrecords.py --tfrecods_path=SL25_tfrecord
"""
import struct
import glob
import tensorflow as tf
import numpy as np
import click
print("Tensorflow version " + tf.__version__)


def _parse_image_function(example_proto):
    """return parse features"""
    image_feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format': tf.io.FixedLenFeature([], tf.string),
        'image/object/bbox/xmin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=None),
        'image/object/bbox/xmax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=None),
        'image/object/bbox/ymin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=None),
        'image/object/bbox/ymax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=None),
        'image/object/class/text': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True, default_value=None),
        'image/object/class/label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=None)
    }
    return tf.io.parse_single_example(example_proto, image_feature_description)


def get_example(tfrecords_path):
    """decode tfrecord file"""
    dataset = tf.data.TFRecordDataset([tfrecords_path])
    parsed_image_dataset = dataset.map(_parse_image_function)
    for image_features in parsed_image_dataset:
        # image
        # image_raw = image_features['image/encoded'].numpy()
        # img = tf.image.decode_image(image_raw)
        image_id = image_features['image/image_id'].numpy()
        text = image_features['image/object/class/text'].numpy()
        label = image_features['image/object/class/label'].numpy()
        print(f'{image_id},{label},{text}')


@click.command(short_help="Verify tfrecords for ml training")
@click.option('--tfrecods_path', help="path to all the tfrecords", required=True, type=str)
def main(tfrecods_path):
    """Verify tfrecords for ml training"""
    f_val = []
    for tfrecod_path in glob.glob(f'{tfrecods_path}/*.tfrecords'):
        print('Reading...' + tfrecod_path)
        f_val.append(tfrecod_path)
        get_example(tfrecod_path)
    n_val_samps = sum([tf.data.TFRecordDataset(f).reduce(np.int64(0), lambda x, _: x + 1).numpy() for f in f_val])
    print('**********************************************************************************')
    print(f'Num items: {n_val_samps}')
    print('**********************************************************************************')


if __name__ == "__main__":
    main()
