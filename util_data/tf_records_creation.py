"""to create TFRecords for ML model training from image chips, label and class id

Author: @developmentseed

Run:
    python3 tf_records_creation.py --tile_path=tiles \
           --csv=tiles_aiaia_sliced_image_nbboxes.csv \
           --csv_class_map=./../config/class_map.csv
           --width=400 \
           --height=400 \
           --output_dir=P1000/SL25_tfrecords \
           --split_in_chunks=True

"""

import os
from os import makedirs, path as op
import io
import json
from collections import namedtuple
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
import click
from smart_open import open

label_map_util.tf = tf.compat.v1
tf.gfile = tf.io.gfile


def split(df, group):
    """Group df by tile_id
    Args:
        df: dataframe
        group (str): argument to group the dataframe
    Returns:
     list of groups
    """
    data = namedtuple('data', ['tile_id', 'object'])
    gb = df.groupby(group)
    return [data(tile_id, gb.get_group(x)) for tile_id, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    """Creates a tf.Example proto from sample wildlife image chips.
    Args:
     encoded wildlife, human or livestock image: The jpg/png encoded data of the image.
    Returns:
     example: The created tf.Example.
    """
    tf_example = None
    image_path = op.join(path, '{}'.format(group.tile_id))
    try:
        with open(image_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)

        width, height = image.size
        filename = group.tile_id.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for _, row in group.object.iterrows():
            xmin, ymin, xmax, ymax = row['bbox']
            xmin = xmin / width
            if xmin < 0.0:
                xmin = 0.0
            elif xmin > 1.0:
                xmin = 1.0
            xmins.append(xmin)

            xmax = xmax / width
            if xmax < 0.0:
                xmax = 0.0
            elif xmax > 1.0:
                xmax = 1.0
            xmaxs.append(xmax)

            ymin = ymin / height
            if ymin < 0.0:
                ymin = 0.0
            elif ymin > 1.0:
                ymin = 1.0
            ymins.append(ymin)

            ymax = ymax / height
            if ymax < 0.0:
                ymax = 0.0
            elif ymax > 1.0:
                ymax = 1.0
            ymaxs.append(ymax)

            classes_text.append(row['label'].encode('utf8'))
            classes.append(int(row['class_id']))
        # print(f'{height},{width},{filename},{filename},-,{image_format},{xmn},{xmx},{ymn},{ymx}')
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))

    except Exception as e:
        print(f"Skipping '{image_path}': {e}")
    return tf_example


def shuffle_split_train_test(df):
    """spliting training data into 70% train, 20% validation and 10% test randomly
    Args:
        df: pandas dataframe in "tile_id, bbox, bbox_norm, label, category"

    Returns:
        df_train, df_val, df_test: train, validation and test dataframe.
    """
    tiles = pd.Series(df.tile_id).unique()
    train_b, val_b = int(len(tiles)*0.7), int(len(tiles)*0.2)
    tile_arr = np.array(list(tiles))
    np.random.shuffle(tile_arr)
    train = tile_arr[:train_b]
    val = tile_arr[train_b:(train_b+val_b)]
    test = tile_arr[(train_b+val_b):]
    df_train = df[df.tile_id.isin(train.tolist())]
    df_val = df[df.tile_id.isin(val.tolist())]
    df_test = df[df.tile_id.isin(test.tolist())]

    return df_train, df_val, df_test


def df_chunks(df, chunk_size):
    """Returns list of dataframe, splited on max size of 500
    """
    list_dfs = [df[i:i+chunk_size] for i in range(0, df.shape[0], chunk_size)]
    return list_dfs


def write_tfrecord_file(df, tile_path, tf_record_path):
    """write tf records file
    Args:
        df: pandas dataframe
        record_file: tfrecord file name
        tile_path: directory to thee chip images
        tf_record_path: path or tthe tf file
    Returns:
        write the tf file
    """
    writer = tf.io.TFRecordWriter(tf_record_path)
    grouped = split(df, 'tile_id')
    for group in grouped:
        tf_example = create_tf_example(group, tile_path)
        if tf_example is not None:
            writer.write(tf_example.SerializeToString())
    writer.close()
    print(f'Successfully created tf records ...{tf_record_path}')


@click.command(short_help="create tfrecords for ml training")
@click.option('--tile_path', help="path to all the image tiles", required=True, type=str)
@click.option('--csv', help="path to a csv that save tile, bbox, label and class_id", required=True, type=str)
@click.option('--output_dir', help="Output path for saving tfrecords", required=True, type=str, default=1000)
@click.option('--width', help="large image's width, e.g. 1000", required=True, type=str, default=1000)
@click.option('--height', help="large image's  height, 1000", required=True, type=str)
@click.option('--split_in_chunks', help="Option to split the tfrecords in chunks", required=False, type=bool, default=False)
def main(tile_path, csv, output_dir, width, height, split_in_chunks):
    """write tfrecords from chip images
    Args:
        tile_path(string): directory to all chips
        csv(string): csv file that contains the chip id and bboxes
        width(integer): size-width of the chips
        height(integer):size-height of the chips
        output_dir(string): output path for saving tfrecords
    Returns:
        (None): written train, val and test tfrcords.
    """
    df = pd.read_csv(csv)
    #################################################
    # Remove label and label_id column and rename group=label and (proup_id or master_group_id)=class_id
    #################################################
    df = df.drop(['label', 'label_id'], axis=1)
    df.rename(columns={'group': 'label'}, inplace=True)

    categories = ['human_activities', 'livestock', 'wildlife', 'master']
    # Output dir for tfrecords
    output_dir = output_dir.strip("/")
    if not op.isdir(output_dir):
        makedirs(output_dir)
    # print('height,width,filename,source_id,encoded,format,xmin,xmax,ymin,ymax,text,label,')
    for category in categories:
        print('=====>' + category)
        # filter df for activities and master
        df_fixed = df.copy()
        if category == 'master':
            df_fixed.rename(columns={'master_group_id': 'class_id'}, inplace=True)
        else:
            df_fixed = df_fixed[df_fixed['category'] == category]
            df_fixed.rename(columns={'group_id': 'class_id'}, inplace=True)

        df_fixed['bbox'] = df_fixed['bbox'].apply((json.loads))
        width, height = int(width), int(height)
        base_name = op.basename(csv).split('_class_id')[0]

        if split_in_chunks:
            list_dfs = df_chunks(df_fixed, 500)
        else:
            list_dfs = [df_fixed]
        for index, chunk_df in enumerate(list_dfs):
            if split_in_chunks:
                sufix = '_'+str(index + 1).zfill(3)
            else:
                sufix = ''
            # name for Tfrecords
            record_file_train = op.join(output_dir, f'train_{base_name}_{category}{sufix}.tfrecords')
            record_file_test = op.join(output_dir, f'test_{base_name}_{category}{sufix}.tfrecords')
            record_file_val = op.join(output_dir, f'val_{base_name}_{category}{sufix}.tfrecords')
            # Split df into train, val, test
            train_df, val_df, test_df = shuffle_split_train_test(chunk_df)
            # train TFRecords Creation
            write_tfrecord_file(train_df, tile_path, record_file_train)

            # Test TFRecords Creation
            write_tfrecord_file(test_df, tile_path, record_file_test)

            # Val TFRecords Creation
            write_tfrecord_file(val_df, tile_path, record_file_val)


if __name__ == "__main__":
    main()
