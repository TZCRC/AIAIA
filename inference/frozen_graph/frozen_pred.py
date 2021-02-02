#!/usr/bin/env python

"""
Inference  with frozen_inference_graph.pb
Author: @developmentseed

Use:
   python frozen_pred.py \
    --frozen_inference_graph=data/frozen_inference_graph.pb \
    --images_path=data/chips/*.jpg \
    --threshold=0.5 \
    --batch_size=5
"""
import os
from os import makedirs, path as op
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import numpy as np
from joblib import Parallel, delayed
import glob
import json
from PIL import Image
from itertools import zip_longest
from numpyencoder import NumpyEncoder
import click
import time
from tqdm import tqdm

start_time = time.time()
# Just disables the warning, doesn't enable AVX/FMA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
print("Tensorflow version " + tf.__version__)

inference_graph="models/livestock/frozen_inference_graph.pb"
detection_graph = tf1.Graph()
with detection_graph.as_default():
    od_graph_def = tf1.GraphDef()
    with tf1.gfile.GFile(inference_graph, "rb") as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf1.import_graph_def(od_graph_def, name="")

def _grouper(iterable, n, fillvalue=None):
    "Itertool recipe to collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def tf1_od_pred(list_images, conf_threshold, frozen_inference_graph):
    """
    funtion to make predictions
    """
    detections = ([])  # contains img name as key and detection info as value (boxes, scores, classes, num)
    with detection_graph.as_default():
        with tf1.Session(graph=detection_graph) as sess:
            # all_tensor_names= [tensor.name for tensor in tf1.get_default_graph().as_graph_def().node]
            # print(all_tensor_names)
            image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")
            detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
            num_detections = detection_graph.get_tensor_by_name("num_detections:0")
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            # image_np_expanded = np.expand_dims(item['img_arr'], axis=0)
            # Actual detection.
            for item_img in list_images:
                (preds_boxes, preds_scores, preds_classes, preds_num,) = sess.run(
                    [
                        detection_boxes,
                        detection_scores,
                        detection_classes,
                        num_detections,
                    ],
                    feed_dict={image_tensor: item_img["img_arr"]},
                )

                #####################################################
                # Filter predictions
                #####################################################
                preds_num = int(preds_num)
                if preds_num > 0:

                    filtered_boxes = []
                    filtered_score = []
                    filtered_classes = []

                    for idx, scores in enumerate(preds_scores):
                        for i, score in enumerate(scores):
                            if score > conf_threshold:
                                filtered_boxes.append(preds_boxes[idx][i])
                                filtered_score.append(score)
                                filtered_classes.append(preds_classes[idx][i])

                    detections.append(
                        {
                            "image_id": item_img["image_id"],
                            "boxes": filtered_boxes,
                            "scores": filtered_score,
                            "classes": filtered_classes,
                            "num": preds_num,
                        }
                    )
                print(f'Predictions for {item_img["image_id"]}: {str(time.time() - start_time)}')

        return detections


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


@click.command(short_help="Testing local tfserving api")
@click.option(
    "-m",
    "--frozen_inference_graph",
    help="frozen inference graph path",
    required=True,
    type=str,
    default="data/ryan_sprint_rcnn_resnet101_serengeti_wildlife_v3_tfs/frozen_inference_graph.pb",
)
@click.option(
    "-s",
    "--images_path",
    help="Path to images",
    required=True,
    type=str,
    default="data/chips/*.jpg",
)
@click.option(
    "-t",
    "--threshold",
    help="threshold to filter the predictions",
    required=True,
    type=float,
    default=0.5,
)
@click.option(
    "-b",
    "--batch_size",
    help="batch size for running inference",
    required=True,
    type=float,
    default=5,
)


def main(frozen_inference_graph, images_path, threshold, batch_size):
    ##########################################
    ### Read images and convert to numpy array
    ##########################################
    images = glob.glob(images_path)
    # images_tensors = []

    for i, img_group in enumerate(_grouper(images, int(batch_size))):
        images_tensors = []
        for batch_img_fname in img_group:
            if batch_img_fname is not None:
                image = Image.open(batch_img_fname)
                image_np = load_image_into_numpy_array(image)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                images_tensors.append({"image_id": batch_img_fname, "img_arr": image_np_expanded})
        print(f"Convert done {len(images)} images to images_tensor: {str(time.time() - start_time)}")

        ##########################################
        ### Inference with frozen_inference_graph.pb in parallel
        ##########################################

        detections = tf1_od_pred(images_tensors, threshold, frozen_inference_graph)
        print(f"Predictions and filters done: {str(time.time() - start_time)}")
        with open(f"result_{i}.json", "w") as f:
            f.write(json.dumps(detections, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
