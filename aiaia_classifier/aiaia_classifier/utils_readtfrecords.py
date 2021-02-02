import os
from typing import Any, Callable, Iterable, List, Optional, Union

import tensorflow as tf
from absl import logging

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


def _augment_helper(image):
    """Augment an image with flipping/brightness changes"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.05)
    image = tf.image.random_contrast(image, 0, 1)
    return image


def _parse_helper(example, n_chan, n_classes, img_dim):
    """"Parse TFExample record containing image and label.""" ""

    example_fmt = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example, example_fmt)

    # Get label, decode
    label = tf.io.parse_tensor(parsed["label"], tf.uint8)

    label = tf.reshape(label, [-1])

    # Get image string, decode
    image = tf.image.decode_image(parsed["image"])
    image = tf.reshape(image, [img_dim, img_dim, 3])

    # change dtype to float32
    image = tf.cast(image, tf.float32)

    # re-scale pixel values between 0 and 1
    image = tf.divide(image, 255)

    return image, label


def parse_and_augment_fn(example, n_chan=3, n_classes=11, img_dim=400):
    """Parse TFExample record containing image and label and then augment image."""
    image, label = _parse_helper(example, n_chan, n_classes, img_dim)
    image = _augment_helper(image)
    return image, label


def parse_fn(example, n_chan=3, n_classes=11, img_dim=400):
    """Parse TFExample record containing image and label."""
    image, label = _parse_helper(example, n_chan, n_classes, img_dim)
    return image, label


def make_dataset(path):
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.shuffle(buffer_size=FLAGS.shuffle_buffer_size)
    dataset = dataset.map(
        map_func=parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.batch(batch_size=FLAGS.batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def _parse_image_function(example_proto):
    image_feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.string),
    }
    return tf.io.parse_single_example(example_proto, image_feature_description)


def get_example(tfrecords_path):
    dataset = tf.data.TFRecordDataset([tfrecords_path])

    parsed_image_dataset = dataset.map(_parse_image_function)

    for image_features in parsed_image_dataset:
        image_raw = image_features["image"].numpy()
        img = tf.image.decode_image(image_raw)

        label = image_features["label"].numpy()
        label = tf.io.parse_tensor(label.numpy(), tf.uint8).numpy()
    return img, label


def country_file_patterns(
    path: str,
    filename_spec: str,
    countries: Iterable[str],
) -> List[str]:
    """Create list of country file patterns for the `get_dataset_feeder` function.

    Parameters
    ----------
    path : str
        Path location of all files to be matched
    filename_spec : str
        String format spec for constructing a country-specific filename, which must
        include exactly one placeholder: ``{country}``
    countries : iterable of str
        One or more countries to match.  To match files for all countries in `path`,
        specify an iterable containing an empty string as a singleton element, along
        with a format spec containing one or more wildcards (``*``).

    Returns
    -------
    list of str
        List of file patterns, one per country specified in `countries`

    Examples
    --------
    Create a single pattern to match all country files in a path:

    >>> country_file_patterns("gs://bucket/path", "train*{country}*.tfrecords", [""])
    ['gs://bucket/path/train**.tfrecords']

    Create a list of patterns, one for each specified country:

    >>> country_file_patterns("gs://bucket/path", "val*{country}*.tfrecords", \
    ["chad", "mali"])
    ['gs://bucket/path/val*chad*.tfrecords', 'gs://bucket/path/val*mali*.tfrecords']
    """
    return [
        os.path.join(path, filename_spec.format(country=country))
        for country in countries
    ]


def get_dataset_feeder(
    file_patterns: Union[str, Iterable[str]],
    *,
    data_map_func: Optional[Callable[[Any], Any]] = None,
    shuffle_buffer_size: Optional[int] = 10000,
    repeat: bool = True,
    n_map_threads: int = 4,
    batch_size: int = 16,
    prefetch_buffer_size: Optional[int] = None,
):
    """Return a TF Dataset for training/evaluating TF Estimators.

    Parameters
    ----------
    file_patterns : str or iterable of str
        One or more glob (wildcard) patterns for selecting files to read.  Note that
        wildcards (`*`) are supported only in the basename portion of the pattern, not
        in the directory portion.
    data_map_func : function or None
        Function to apply to each element of the dataset.
    shuffle_buffer_size : int or None
        Buffer size to use for shuffling the dataset.  A size of ``None`` or zero causes
        shuffling to be skipped.
    repeat : bool
    n_map_threads : int
    batch_size : int
    prefetch_buffer_size : int or None

    Returns
    -------
    TFRecordDataset
        Dataset constructed from files matching the specified patterns, with elements
        mapped (if `data_map_func` is not ``None``), and shuffled, if
        `shuffle_buffer_size` is an integer greater than 0
    """
    logging.info(f"Searching for files matching the patterns {file_patterns}")
    files = tf.io.matching_files(file_patterns)
    logging.info(f"Reading records from the files {files}")
    ds = tf.data.TFRecordDataset(tf.data.Dataset.from_tensor_slices(files))

    if shuffle_buffer_size:
        ds = ds.shuffle(shuffle_buffer_size)
    if repeat:
        ds = ds.repeat()
    if data_map_func:
        ds = ds.map(map_func=data_map_func, num_parallel_calls=n_map_threads)

    ds = ds.batch(batch_size=batch_size)

    if prefetch_buffer_size:
        ds = ds.prefetch(buffer_size=prefetch_buffer_size)

    return ds
