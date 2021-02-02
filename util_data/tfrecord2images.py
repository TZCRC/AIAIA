"""
Script to convert tf records to images and to draw bbox over images for labels' sanity check

Author: @developmentseed

Run:

python tfrecord2images.py \
    --tfrecords_path=aiaia_od/training_data_aiaia/SL25_tfrecord/*.tfrecords  \
    --output_dir=abc
"""
import os
from os import path as op
import glob
import tensorflow as tf
import numpy as np
from object_detection.utils import dataset_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import pandas as pd
import click
import itertools
from PIL import Image, ImageDraw
from PIL import ImageColor
import glob

print("Tensorflow version " + tf.__version__)


def _parse_image_function(example_proto):
    """return parse features"""
    image_feature_description = {
        'image/height': tf.io.FixedLenFeature([], tf.int64),
        'image/width': tf.io.FixedLenFeature([], tf.int64),
        'image/filename': tf.io.FixedLenFeature([], tf.string),
        'image/source_id': tf.io.FixedLenFeature([], tf.string),
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
    items = []
    for image_features in parsed_image_dataset:
        image_id = image_features["image/filename"].numpy()
        image_raw = image_features["image/encoded"].numpy()
        img = tf.image.decode_image(image_raw)
        label = image_features["image/object/class/label"].numpy()
        cls_name = image_features["image/object/class/text"].numpy()
        ymin = image_features["image/object/bbox/ymin"].numpy()
        xmin = image_features["image/object/bbox/xmin"].numpy()
        ymax = image_features["image/object/bbox/ymax"].numpy()
        xmax = image_features["image/object/bbox/xmax"].numpy()
        bboxes = [[xmin[i], ymin[i], xmax[i], ymax[i]] for i in range(len(label))]
        labels = [label[i] for i in range(len(label))]
        cls_names = [cls_name[i] for i in range(len(cls_name))]
        items.append(
            {
                "image_id": image_id,
                "img_arr": img,
                "labels": labels,
                "class_name": cls_name,
                "bboxes": bboxes,
            }
        )

    return items


def save_image(item, output_dir):

    image_np = item["img_arr"].numpy()
    image_id = str(item["image_id"].decode('utf-8')).strip("''")
    bboxes = item["bboxes"]
    gt_labels = item["labels"]
    class_name = item["class_name"]

    ############################
    # Draw bbox
    ############################
    img = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(img)
    image_path = op.join(output_dir, str(image_id))
    for i, bbox in enumerate(bboxes):
        bbox = [bbox[i] * 400 for i in range(len(bbox))]
        xmin, ymin, xmax, ymax = bbox
        # xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        draw.rectangle(bbox, outline="#0000ff")
        x_label = xmin + (xmax - xmin) / 2
        draw.text(
            (x_label - 15, ymax),
            text=str(class_name[i].decode('utf-8')),
            fill="red",
            align="right",
        )

        draw.text(
            (xmin, ymin),
            text=str(f"({xmin}, {ymin})"),
            fill="yellow",
            align="left",
        )

        draw.text(
            (xmax, ymax),
            text=str(f"({xmax}, {ymax})"),
            fill="yellow",
            align="left",
        )
    img.save(image_path, "JPEG")


@click.command(short_help="Script to convert tf records to images and to draw bbox over images for labels' sanity check")
@click.option(
    "--tfrecords_path",
    help="Path for tfreecord",
    required=True,
    type=str,
)
@click.option(
    "--output_dir",
    help="Output path for saving images",
    required=True,
    type=str,
)
def main(tfrecords_path, output_dir):

    for tfrecod_path in glob.glob(tfrecords_path):
        print(tfrecod_path)
        tfrecord_filename = os.path.splitext(os.path.basename(tfrecod_path))[0]
        sub_dir = op.join(output_dir,'')
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        items = get_example(tfrecod_path)
        for item in items:
            save_image(item, sub_dir)
if __name__ == "__main__":
    main()
