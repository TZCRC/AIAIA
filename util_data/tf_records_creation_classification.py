"""to create TFRecords for ML classification model training from image chips, label and class id

Author: @developmentseed

Run:
    python3 tf_records_creation_classification.py \
        --tile_path=data/P400_v2/ \
        --csv_files=data/csv/*_class_id.csv \
        --output_dir=data/classification_training_tfrecords/ \
        --output_csv=data/csv/classification_training_tfrecords.csv

"""

import os
from os import makedirs, path
import io
import json
from collections import namedtuple
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image

import click
from smart_open import open
import glob
import random


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = (
            value.numpy()
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_feature_label(value):
    """Returns a bytes_list from classes in int """

    label = tf.convert_to_tensor([0, 1] if value else [1, 0], dtype=tf.uint8)
    label = tf.io.serialize_tensor(label)

    if isinstance(label, type(tf.constant(0))):
        label = (
            label.numpy()
        )  # BytesList won't unpack a string from an EagerTensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def df_chunks(df, chunk_size):
    """Returns list of dataframe, splited on max size of 500"""
    list_dfs = [
        df[i : (i + chunk_size)] for i in range(0, df.shape[0], chunk_size)
    ]
    return list_dfs


def shuffle_split_train_test(df):
    """spliting training data into 70% train, 20% validation and 10% test randomly
    Args:
        df: pandas dataframe in "chip , label"
    Returns:
        df_train, df_val, df_test: train, validation and test dataframe.
    """
    tiles = pd.Series(df.chip).unique()
    train_b, val_b = int(len(tiles) * 0.7), int(len(tiles) * 0.2)
    tile_arr = np.array(list(tiles))
    np.random.shuffle(tile_arr)
    train = tile_arr[:train_b]
    val = tile_arr[train_b : (train_b + val_b)]
    test = tile_arr[(train_b + val_b) :]
    df_train = df[df.chip.isin(train.tolist())]
    df_val = df[df.chip.isin(val.tolist())]
    df_test = df[df.chip.isin(test.tolist())]
    return df_train, df_val, df_test


def map_labels(df):
    """customize label classes for objects"""
    image_labels = dict(zip(df.chip, df.label))
    return image_labels


def image_example(image_string, label, image_shape):
    feature = {
        "height": _int64_feature(image_shape[0]),
        "width": _int64_feature(image_shape[1]),
        "depth": _int64_feature(image_shape[2]),
        "label": _bytes_feature_label(label),
        "image": _bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tf_file(df, record_file, labels, st_dir, image_shape, description):
    """write tf records files
    Args:
        df: pandas dataframe
        record_file: tfrecord file name
        labels: dictionary of labels
        st_dir: directory to all supertiles
        image_shape: shape of the images
        description: description for printing on the outputs
    Returns:
        write the tf file
    """
    print("#" * 60)
    print(f"{len(df)} {description}")
    print(f"{len(df[df['label']==1])} are objects")
    print(f"{len(df[df['label']==0])} are not objects")
    print("#" * 60)

    with tf.io.TFRecordWriter(record_file) as writer:
        for filename, label in labels.items():
            if filename is None:
                continue
            try:
                with open(filename, "rb") as image_file:
                    image = image_file.read()
            except Exception as e:
                print(f"Skipping '{filename}': {e}")
                continue

            tf_example = image_example(image, label, image_shape)
            writer.write(tf_example.SerializeToString())


def write_tfrecords(df, st_dir, chunk_size, output):
    """write tfrecords for classification training

    Args:
        df: dataframe. chip, label
        st_dir(string): directory to all supertiles
        chunk_size(int): number of features to split the df and to be considered in the
            TFrecords
        super_tile(bool): if it's supertile

    Returns:
        (None): written train, val and test tfrcords.
    """

    chunk_dfs = df_chunks(df, chunk_size)
    base_name = "aiaia"
    image_shape = (400, 400, 3)

    for index, chunk_df in enumerate(chunk_dfs):
        suffix = str(index + 1).zfill(3)
        train_df, val_df, test_df = shuffle_split_train_test(chunk_df)
        # train
        record_file = os.path.join(
            output, f"train_{base_name}_{suffix}.tfrecords"
        )
        train_labels = map_labels(train_df)
        write_tf_file(
            train_df,
            record_file,
            train_labels,
            st_dir,
            image_shape,
            "samples as training",
        )

        # validation
        record_file = os.path.join(
            output, f"val_{base_name}_{suffix}.tfrecords"
        )
        val_labels = map_labels(val_df)
        write_tf_file(
            val_df,
            record_file,
            val_labels,
            st_dir,
            image_shape,
            "samples as validation",
        )

        # test
        record_file = os.path.join(
            output, f"test_{base_name}_{suffix}.tfrecords"
        )
        test_labels = map_labels(test_df)
        write_tf_file(
            test_df,
            record_file,
            test_labels,
            st_dir,
            image_shape,
            "samples as testing",
        )

    print("Finished writing TFRecords.")


@click.command(short_help="create tfrecords for classification training")
@click.option(
    "--tile_path",
    help="path to all the image tiles",
    required=True,
    type=str,
    default="data/P400_v2/",
)
@click.option(
    "--path_csv_files",
    help="path csv files",
    required=True,
    type=str,
    default="data/csv/*_class_id.csv",
)
@click.option(
    "--output_dir",
    help="Output path for saving tfrecords",
    required=True,
    type=str,
    default="data/classification_training_tfrecords",
)
@click.option(
    "--output_csv",
    help="Output csv path",
    required=True,
    type=str,
    default="data/csv/classification_training_tfrecords.csv",
)
def main(tile_path, path_csv_files, output_dir, output_csv):

    if not path.isdir(output_dir):
        makedirs(output_dir)

    # # #################################################
    # # # Filter unique chips
    # # #################################################
    csv_files = glob.glob(path_csv_files)
    frames = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        prefix, _, _ = csv_file.split("/")[2].split("_")
        prefix = tile_path + prefix + "_tiles/"
        df["tile_id"] = prefix + df["tile_id"].astype(str)
        frames.append(df)

    df = pd.concat(frames)
    # Delete boma labels
    df = df.drop(df["label"] == "boma")
    print(f"Total rows {df.shape[0]}")
    df = df.drop_duplicates(subset=["tile_id"])
    print(f"Total unique rows {df.shape[0]}")
    tiles_yes = list(df["tile_id"].to_numpy())

    # # # #################################################
    # # # # Loading randomly 20k chips and make sure that no repeats with the chips with objects
    # # # #################################################

    chips = glob.glob(tile_path + "/*/*.jpg")
    sampled_chips = random.sample(chips, 20000)
    result = list(set(sampled_chips) - set(tiles_yes))
    # Filter 14k chips
    tiles_yes = random.sample(tiles_yes, 7000)
    tiles_no = random.sample(result, 7000)
    intersection = list(set(tiles_yes) & set(tiles_no))
    if len(intersection) == 0:
        result_list = []
        for tile in tiles_yes:
            result_list.append({"chip": tile, "label": 1})
        for tile in tiles_no:
            result_list.append({"chip": tile, "label": 0})
        random.shuffle(result_list)
        df_result = pd.DataFrame(result_list)
        # save the data in csv file
        df_result.to_csv(output_csv, index=False)
        # # # #################################################
        # # # # Writte tfrecords
        # # # #################################################
        write_tfrecords(df_result, tile_path, 500, output_dir)


if __name__ == "__main__":
    main()
