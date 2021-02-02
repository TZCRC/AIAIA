#!/usr/bin/env python
# coding: utf-8

#### Citations
#
# https://github.com/tensorflow/models/tree/master/research/object_detection
# "Speed/accuracy trade-offs for modern convolutional object detectors."
# Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z,
# Song Y, Guadarrama S, Murphy K, CVPR 2017
#
# https://github.com/rafaelpadilla/Object-Detection-Metrics
# @INPROCEEDINGS {padillaCITE2020,
#     author    = {R. {Padilla} and S. L. {Netto} and E. A. B. {da Silva}},
#     title     = {A Survey on Performance Metrics for Object-Detection Algorithms},
#     booktitle = {2020 International Conference on Systems, Signals and Image Processing (IWSSIP)},
#     year      = {2020},
#     pages     = {237-242},}
#

# ## Loading in TFRecords from GCS bucket
import os
from os import path as op
import tensorflow as tf  # version 2
import tensorflow.compat.v1 as tf1
from object_detection.utils import dataset_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import glob
import matplotlib.pyplot as plt
import click
import seaborn as sn


# Just disables the warning, doesn't enable AVX/FMA
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
print("Tensorflow version " + tf.__version__)


def _parse_image_function(example_proto):
    """return parse features"""
    image_feature_description = {
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/filename": tf.io.FixedLenFeature([], tf.string),
        "image/source_id": tf.io.FixedLenFeature([], tf.string),
        "image/encoded": tf.io.FixedLenFeature([], tf.string),
        "image/format": tf.io.FixedLenFeature([], tf.string),
        "image/object/bbox/xmin": tf.io.FixedLenSequenceFeature(
            [], tf.float32, allow_missing=True, default_value=None
        ),
        "image/object/bbox/xmax": tf.io.FixedLenSequenceFeature(
            [], tf.float32, allow_missing=True, default_value=None
        ),
        "image/object/bbox/ymin": tf.io.FixedLenSequenceFeature(
            [], tf.float32, allow_missing=True, default_value=None
        ),
        "image/object/bbox/ymax": tf.io.FixedLenSequenceFeature(
            [], tf.float32, allow_missing=True, default_value=None
        ),
        "image/object/class/text": tf.io.FixedLenSequenceFeature(
            [], tf.string, allow_missing=True, default_value=None
        ),
        "image/object/class/label": tf.io.FixedLenSequenceFeature(
            [], tf.int64, allow_missing=True, default_value=None
        ),
    }
    return tf.io.parse_single_example(example_proto, image_feature_description)


def get_example(tfrecords_path):
    """decode tfrecord file, returns items"""
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
        bboxes = [
            [xmin[i], ymin[i], xmax[i], ymax[i]] for i in range(len(label))
        ]
        labels = [label[i] for i in range(len(label))]
        cls_names = [cls_name[i] for i in range(len(cls_name))]
        items.append(
            {
                "image_id": image_id,
                "img_arr": img,
                "labels": labels,
                "class_name": cls_names,
                "bboxes": bboxes,
            }
        )
    return items


def get_example_no_img(tfrecords_path):
    """decode tfrecord file, returns items"""
    dataset = tf.data.TFRecordDataset([tfrecords_path])
    parsed_image_dataset = dataset.map(_parse_image_function)
    items = []
    for image_features in parsed_image_dataset:
        image_id = image_features["image/filename"].numpy()
        label = image_features["image/object/class/label"].numpy()
        cls_name = image_features["image/object/class/text"].numpy()
        ymin = image_features["image/object/bbox/ymin"].numpy()
        xmin = image_features["image/object/bbox/xmin"].numpy()
        ymax = image_features["image/object/bbox/ymax"].numpy()
        xmax = image_features["image/object/bbox/xmax"].numpy()
        bboxes = [
            [xmin[i], ymin[i], xmax[i], ymax[i]] for i in range(len(label))
        ]
        labels = [label[i] for i in range(len(label))]
        cls_names = [cls_name[i] for i in range(len(cls_name))]
        items.append(
            {
                "image_id": image_id,
                "labels": labels,
                "class_name": cls_names,
                "bboxes": bboxes,
            }
        )
    return items


def tf1_od_pred(test_img_dict_lst, detection_graph):
    """
    Runs inference with a frozen graph on images and returns list of dicts, with image_id.
    Args:
        test_img_dict_lst (list): list of dicts containing the image and image gt metadata, read from the tfrecord by get_example().
        detection_graph (str): the loaded frozen graph
    Returns:
        det_dicts (list): A list of dictionaries containing detection data, including bounding boxes and class scores
    """

    det_dicts = (
        []
    )  # contains img name as key and detection info as value (boxes, scores, classes, num)
    with detection_graph.as_default():
        with tf1.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name(
                "detection_boxes:0"
            )
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name(
                "detection_scores:0"
            )
            detection_classes = detection_graph.get_tensor_by_name(
                "detection_classes:0"
            )
            num_detections = detection_graph.get_tensor_by_name(
                "num_detections:0"
            )
            for item in test_img_dict_lst:
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(item["img_arr"], axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [
                        detection_boxes,
                        detection_scores,
                        detection_classes,
                        num_detections,
                    ],
                    feed_dict={image_tensor: image_np_expanded},
                )

                det_dicts.append(
                    {
                        "image_id": item["image_id"],
                        "boxes": boxes,
                        "scores": scores,
                        "classes": classes,
                        "num": num,
                    }
                )
        return det_dicts


def filter_detections(det, conf_threshold):
    # remove artifact of no detection
    if det["num"][0] == 0.0:
        det["scores"] = []
        det["boxes"] = []
        det["classes"] = []
    # remove low scores
    high_score_mask = np.array(det["scores"]) > conf_threshold
    det["scores"] = np.array(det["scores"])[high_score_mask]
    det["boxes"] = np.array(det["boxes"])[high_score_mask]
    det["classes"] = np.array(det["classes"])[high_score_mask]
    return det


def filter_detections_and_concat_boxes(all_gt, det_dicts, conf_threshold):
    """
    Filters detections and groundtruth by a confidence score threshold and removes empty detections.
    Args:
        all_gt (list): all groundtruth dictionaries
        det_dicts (list): all detection dictionaries
        conf_threshold (float): all values below this will not be included as valid detections that are used to generate metrics
    Returns:
        A filtered lists of dictionaries, as well as lists of bounding boxes and classes extracted from these dictionaries

    """
    all_det_scores = []
    all_det_img_names = []
    all_det_boxes = []
    all_gt_img_names = []
    all_gt_boxes = []
    all_gt_classes = []
    all_det_classes = []
    for gt, det in zip(all_gt, det_dicts):
        det = filter_detections(det, conf_threshold)
        if len(det["boxes"]) > 0:
            all_det_boxes.append(det["boxes"])
            all_det_classes.extend(det["classes"])
            all_det_scores.extend(det["scores"])
            all_det_img_names.append(det["image_id"])
        all_gt_boxes.append(np.array(gt["bboxes"]))
        all_gt_classes.extend(gt["labels"])
        all_gt_img_names.append(gt["image_id"])
    all_gt_boxes = np.concatenate(all_gt_boxes)
    all_det_boxes = np.concatenate(all_det_boxes)
    all_gt_classes = np.array(all_gt_classes).astype(int)
    all_det_classes = np.array(all_det_classes).astype(int)
    all_det_scores = np.array(all_det_scores).astype(float)
    return (
        all_gt,
        det_dicts,
        all_gt_boxes,
        all_det_boxes,
        all_gt_classes,
        all_det_classes,
        all_det_scores,
        all_det_img_names,
        all_gt_img_names,
    )


def compute_iou(groundtruth_box, detection_box):
    """
    compute IOU score by compare ground truth bbox and predicted bbox
    Args:
        groundtruth_box: ground truth bbox in [x0, y0, x1, y1]
        detection_box: predicted truth bbox in [y0, x0, y1, x1]
    Returns:
        iou: IOU score between ground truth and detected bboxes
    """
    g_xmin, g_ymin, g_xmax, g_ymax = tuple(groundtruth_box)
    d_ymin, d_xmin, d_ymax, d_xmax = tuple(detection_box)

    x_left = max(g_xmin, d_xmin)
    y_top = max(g_ymin, d_ymin)
    x_right = min(g_xmax, d_xmax)
    y_bottom = min(g_ymax, d_ymax)

    boxGArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)

    if x_right < x_left or y_bottom < y_top:
        return 0, boxGArea

    intersection = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    boxDArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)
    iou = intersection / float(boxGArea + boxDArea - intersection)
    return iou, boxGArea


def get_box_matches(
    groundtruth_boxes, detection_boxes, detection_scores, iou_threshold
):
    """
    Returns pred list and gt list indices for the box matches, the iou,
    and the groundtruth box area to examine size effect on accuracy.

    Args:
        groundtruth_boxes: list of ground truth bbox in [x0, y0, x1, y1]
        detection_boxes: list of predicted truth bbox in [y0, x0, y1, x1]
        IOU_THRESHOLD: threshold used to consider a detection a valid overlap and possible true positive
    Returns:
        iou: IOU score between ground truth and detected bboxes
    """

    matches = []

    for i in range(len(groundtruth_boxes)):
        for j in range(len(detection_boxes)):
            iou, gt_box_area = compute_iou(
                groundtruth_boxes[i], detection_boxes[j]
            )

            if iou > iou_threshold:
                matches.append([i, j, iou, detection_scores[j], gt_box_area])

    matches = np.array(matches)
    if matches.shape[0] > 0:
        # Sort list of matches by descending IOU so we can remove duplicate detections
        # while keeping the highest IOU entry.
        matches = matches[matches[:, 2].argsort()[::-1][: len(matches)]]

        # Remove duplicate detections from the list.
        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

        # Sort the list again by descending confidence score first, then IOU.
        # This deals with cases where a gt detection has multiple high IOU
        # detections of different classes, with one correct, higher confidence
        # detection. Removing duplicates doesn't preserve our previous sort.

        matches = matches[np.lexsort((matches[:, 2], matches[:, 3]))[::-1]]
        # Remove duplicate ground truths from the list.
        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

    return matches


def compute_confusion_matrix(matches, gt_classes, det_classes, num_categories):
    """Computes a confusion matrix to count true positives, false negatives, false positives.

    This iterates over the groundtruth and detection indices representing
    bounding boxes and their classes. Matches are identified when IOU is
    above the threshold.Each match in `matches` will count as a true positive
    or misidentified positive in a confusion matrix cell. If indices are not
    in `matches` they are then counted as either false negative or false positive.

    Worth noting, many detections ascribed as false positives actually have
    substantial overlap with a groundtruth detection, but not as much overlap as the
    matching true positive. In order to not double count objects, these extra
    detections need to be set as false positives.

    Args:
        matches (list): [description]
        gt_classes (list): [description]
        det_classes (list): [description]
        num_categories (int, optional): The number of categories.

    Returns:
        [type]: [description]
    """

    gt_classes_sorted = []
    det_classes_sorted = []
    confusion_matrix = np.zeros(shape=(num_categories + 1, num_categories + 1))
    for i in range(len(gt_classes)):
        if matches.shape[0] > 0 and matches[matches[:, 0] == i].shape[0] == 1:
            # match identified, count a true positive or a misidentification
            detection_class_cm_i = int(
                det_classes[int(matches[matches[:, 0] == i, 1][0])] - 1
            )  # index along cm axis and the class label associated with the match
            groundtruth_class_cm_i = int(gt_classes[i] - 1)
        else:
            # detection is background but groundtruth is another class, false negative
            detection_class_cm_i = int(confusion_matrix.shape[1] - 1)
            groundtruth_class_cm_i = int(gt_classes[i] - 1)

        gt_classes_sorted.append(groundtruth_class_cm_i)
        det_classes_sorted.append(detection_class_cm_i)
        confusion_matrix[groundtruth_class_cm_i][detection_class_cm_i] += 1

    for i in range(len(det_classes)):
        # catches case where some detections have matches and when there are no matches but there are gt
        if (
            matches.shape[0] > 0 and matches[matches[:, 1] == i].shape[0] == 0
        ) or matches.size == 0:
            detection_class_cm_i = int(det_classes[i] - 1)
            groundtruth_class_cm_i = int(confusion_matrix.shape[0] - 1)

            gt_classes_sorted.append(groundtruth_class_cm_i)
            det_classes_sorted.append(detection_class_cm_i)
            confusion_matrix[groundtruth_class_cm_i][detection_class_cm_i] += 1

    assert len(gt_classes_sorted) == len(det_classes_sorted)

    return confusion_matrix, gt_classes_sorted, det_classes_sorted


def plot_cm_sn(
    cm,
    names,
    outpath,
    title,
    iou_threshold,
    conf_threshold,
    normalize,
    norm_axis="predicted",
    cmap="Blues",
):
    num_classes = len(names)
    cm_copy = cm.copy()
    if normalize:
        if norm_axis == "predicted":
            cm_copy = cm / (cm.sum(0).reshape(1, num_classes + 1) + 1e-6)
        elif norm_axis == "true":
            cm_copy = cm / (cm.sum(1).reshape(num_classes + 1, 1) + 1e-6)
    cm_copy[cm_copy < 0.005] = np.nan  # don't annotate (would appear as 0.00)
    # reorder confusion matrix by diagonal
    diag = np.diag(cm_copy)  # get diagonal
    idx = np.argsort(
        diag
    )  # get all indicies of diag ordered, but true values sorted wrong
    idx_nonnan = np.argwhere(~np.isnan(diag))  # get nonnan indices
    nonnanlen = len(idx_nonnan)
    idx[0:nonnanlen] = np.concatenate(
        idx_nonnan, axis=0
    )  # replace non nan indices with sorted
    yticklabels = np.array(names + ["False Negative, \nonly groundtruth"])
    xticklabels = np.array(names + ["False Positive, \nonly detection"])
    cm_copy = cm_copy[idx, :][:, idx]
    yticklabels = yticklabels[idx]
    xticklabels = xticklabels[idx]
    # plot
    fig = plt.figure(figsize=(12, 9))
    sn.set(font_scale=1.0 if num_classes < 50 else 0.8)  # for label size
    labels = (0 < len(names) < 99) and len(
        names
    ) == num_classes  # apply names to ticklabels
    sn.heatmap(
        cm_copy,
        annot=num_classes < 30,
        annot_kws={"size": 8},
        cmap=cmap,
        fmt=".2f",
        square=True,
        yticklabels=list(yticklabels)
        if labels
        else "auto",  # switched ticklabels
        xticklabels=list(xticklabels) if labels else "auto",
    ).set_facecolor((1, 1, 1))
    fig.axes[0].set_xlabel("True", fontsize=16)
    fig.axes[0].set_ylabel(
        "Predicted",
        fontsize=16,
    )
    fig.axes[0].set_title(
        title
        + f"\n {iou_threshold} IOU and  {conf_threshold} Confidence Score Thresholds",
        fontsize=19,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.yticks(rotation=0)
    plt.close()


def cm_from_img_tuple(img_detection_tuple, class_names, IOU_thresh=0.5):
    """Computes confusion matrix from a single sample of predictions and groundtruth.

    Args:
        img_detection_tuple (tuple): Tuple with two dictionaries, one for
            predictions and one for groundtruth.
        IOU_thresh (float):  The intersection over union threshold used to
            determine a match.

    Returns:
        np.array: The confusion matrix in numpy array format.
    """
    img_gt_dict = img_detection_tuple[0]
    det_dict = img_detection_tuple[1]
    assert det_dict["image_id"] == img_gt_dict["image_id"]
    gt_bboxes = img_gt_dict["bboxes"]
    gt_labels = img_gt_dict["labels"]
    pred_scores = det_dict["scores"]
    pred_bboxes = det_dict["boxes"]
    pred_labels = det_dict["classes"].astype(int)

    # # Getting image specific counts of TP, FP, FN
    box_matches = get_box_matches(
        gt_bboxes, pred_bboxes, pred_scores, IOU_thresh
    )
    # y_true, y_pred for scikit_learn format to compute metrics
    cm, y_true, y_pred = compute_confusion_matrix(
        box_matches, gt_labels, pred_labels, len(class_names)
    )

    return cm


def save_image(
    img_detection_tuple,
    outdir,
    class_names,
    iou_threshold,
    conf_threshold,
    cm=None,
):
    """Saves images with bounding boxes for prediction (blue) and groundtruth (red).

    Args:
        img_detection_tuple (tuple): Tuple with two dictionaries, one for predictions
            and one for groundtruth.
        outdir (str): where to save the images
        class_names (list): list of class names. must be supplied in order that matches
            both detection and groundtruth numerical class IDs
        cm (np.array, optional): The confusion matrix corresponding to img_detection_tuple.
            Defaults to None.
    """

    img_gt_dict = img_detection_tuple[0]
    det_dict = img_detection_tuple[1]
    assert det_dict["image_id"] == img_gt_dict["image_id"]
    image_np = img_gt_dict["img_arr"].numpy()
    image_id = str(img_gt_dict["image_id"].decode("utf-8")).strip("''")
    gt_bboxes = img_gt_dict["bboxes"]
    gt_class_name = img_gt_dict["class_name"]

    pred_bboxes = det_dict["boxes"]
    pred_labels = det_dict["classes"]
    pred_scores = det_dict["scores"]

    if cm is not None:
        cmname = image_id[:-4] + "_cm" + ".png"
        plot_cm_sn(
            cm,
            class_names,
            outpath=op.join(outdir, cmname),
            title=f"{image_id[:-4]} Only Confusion Matrix",
            normalize=False,
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold,
        )

    ############################
    # Draw bbox
    ############################
    img = Image.fromarray(image_np).convert("RGB")
    draw = ImageDraw.Draw(img)
    image_path = op.join(outdir, str(image_id))
    for i, gtbbox in enumerate(gt_bboxes):
        gtbbox = [gtbbox[i] * 400 for i in range(len(gtbbox))]
        xmin, ymin, xmax, ymax = gtbbox
        draw.rectangle(gtbbox, outline="#ff0000")
        x_label = xmin + (xmax - xmin) / 2
        draw.text(
            (x_label - 15, ymax),
            text=str(gt_class_name[i].decode("utf-8")),
            fill="red",
            align="right",
        )
    for i, pred_bbox in enumerate(pred_bboxes):
        pred_bbox = [pred_bbox[i] * 400 for i in range(len(pred_bbox))]
        ymin, xmin, ymax, xmax = pred_bbox
        pred_bbox = [xmin, ymin, xmax, ymax]
        draw.rectangle(pred_bbox, outline="#0000ff")
        x_label = xmin + (xmax - xmin) / 2
        class_i = int(pred_labels[i])
        draw.text(
            (x_label - 15, ymax),
            text=f"{class_names[class_i-1]} {np.round(pred_scores[i],decimals=3).astype('|S4').decode('utf-8')}",
            fill="blue",
            align="right",
        )

    img.save(image_path, "JPEG")
    return cmname


def save_metrics(confusion_matrix, categories, output_path, iou_threshold):
    """
    Write a cvs that saves recall, precision, f1 and map score
    Args:
        confusion_matrix: computed confusion matrix for classes
        categories: classes in list;
        output_path: output path for the csv
    Returns:
        (None): saved csv
    """
    print("\nConfusion Matrix:")
    print(confusion_matrix, "\n")
    print(f"Confusion Matrix Shape: {confusion_matrix.shape}")

    labels = [categories[i] for i in range(len(categories))]
    results = []

    for i in range(len(categories)):
        name = categories[i]
        total_target = np.sum(confusion_matrix[:, i])
        total_predicted = np.sum(confusion_matrix[i, :])
        if total_target == 0:
            recall = 0
        else:
            recall = float(confusion_matrix[i, i] / total_target)
            # print(f'recalls are {recall}')
        if total_predicted == 0:
            precision = 0
        else:
            precision = float(confusion_matrix[i, i] / total_predicted)
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0
        prec_at_rec = []
        for recall_level in np.linspace(0.0, 1.0, 11):

            if recall >= recall_level:
                prec_at_rec.append(precision)
        prec_at_rec = np.array(prec_at_rec)
        avg_prec = np.mean(prec_at_rec)

        results.append(
            {
                "category": name,
                f"precision_@{ iou_threshold}IOU": precision,
                f"recall_@{ iou_threshold}IOU": recall,
                f"map_@{ iou_threshold}IOU": avg_prec,
                f"f1_@{ iou_threshold}IOU": f1,
            }
        )
    df = pd.DataFrame(results)
    df.to_csv(output_path)


@click.command()
@click.option(
    "--tfrecords_folder",
    default="./training_data_aiaia_p400",
    help="The folder containing the subfolders that contain tfrecords.",
)
@click.option(
    "--outdir",
    default="./outputs",
    help="Where to save output images with bounding boxes drawn, metrics, and plots.",
)
@click.option(
    "--model_rel_path",
    default="./frozen_inference_graph.pb",
    help="The path to the folder containing the frozen graph .pb file.",
)
@click.option(
    "--class_names",
    "-cn",
    multiple=True,
    default=[
        "buffalo",
        "dark_coloured_large",
        "elephant",
        "giraffe",
        "hippopotamus",
        "light_coloured_large",
        "smaller_ungulates",
        "warthog",
        "zebra",
    ],
    help="The class names that match the order of the class IDs from the prediction output and groundtruth. Use like -cn buffalo -cn dark_coloured_large -cn elephant etc. Order matters and should match the order defined in the class_map.csv.",
)
@click.option(
    "--model_type",
    default="wildlife",
    type=click.Choice(
        ["human_activities", "wildlife", "livestock"], case_sensitive=True
    ),
    help="The model type used to filter tfrecords in subfolders to run evaluation on different detection problems. These include human_activities, wildlife, and livestock.",
)
@click.option(
    "--iou_threshold",
    default=0.5,
    type=float,
    help="Threshold to set boxes with low overlap as not potential true positives. Defaults to .5",
)
@click.option(
    "--conf_threshold",
    default=0.5,
    type=float,
    help="Threshold to throw away low confidence detections. Defaults to .5",
)
@click.option(
    "--save_individual_images",
    default=True,
    type=bool,
    help="Whether to save individual images with bounding boxes drawn. Useful for debugging and inspecting model results. Defaults to True",
)
def run_eval(
    tfrecords_folder,
    outdir,
    model_rel_path,
    class_names,
    model_type,
    iou_threshold,
    conf_threshold,
    save_individual_images,
):
    """Computes metrics for TFrecords containing test images and groundtruth.

    Also saves out groundtruth and prediction boxes drawn on images.
    A frozen graph model must be supplied for predictions. Computes, plots and
    saves confusion matrix and a csv with metrics, including total f1 score.
    """
    print(f"Starting evaluation for model: {model_type}")
    # Reading all images and groundtruth
    class_names = list(class_names)  # it initially gets parsed as a tuple
    all_gt = []
    all_imgs_and_gt = []
    tfrecord_regex = f"test*{model_type}*.tfrecords"
    # list tfrecord folders
    for tfrecords_subfolder in glob.glob(tfrecords_folder + "/*/"):
        for tfrecord_path in glob.glob(
            op.join(tfrecords_subfolder, tfrecord_regex)
        ):
            sub_dir = op.join(outdir, "")
            if not op.exists(sub_dir):
                os.makedirs(sub_dir)
            items = get_example(tfrecord_path)
            items_no_imgs = get_example_no_img(tfrecord_path)
            all_gt.extend(items_no_imgs)
            all_imgs_and_gt.extend(items)
    print("TFRecords opened")
    # Testing the tf model on subset of test set, all_imgs_and_gt
    model_path = op.join(os.getcwd(), model_rel_path)
    detection_graph = tf1.Graph()
    with detection_graph.as_default():
        od_graph_def = tf1.GraphDef()
        with tf1.gfile.GFile(model_path, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf1.import_graph_def(od_graph_def, name="")
    det_dicts = tf1_od_pred(all_imgs_and_gt, detection_graph)
    print("Inference complete")
    (
        all_gt,
        det_dicts_filtered,
        all_gt_boxes,
        all_detection_boxes,
        all_gt_classes,
        all_det_classes,
        all_det_scores,
        all_detection_img_names,
        all_gt_img_names,
    ) = filter_detections_and_concat_boxes(
        all_gt, det_dicts, conf_threshold=conf_threshold
    )
    print("Detections filtered by confidence")
    # save out images with both boxes drawn for sanity check
    if save_individual_images:
        test_imgs_dir = op.join(outdir, "test_imgs_with_boxes")
        if not op.exists(test_imgs_dir):
            os.makedirs(test_imgs_dir)

    all_imgs_gt_det = list(zip(all_imgs_and_gt, det_dicts_filtered))
    cms = []
    for item in all_imgs_gt_det:
        cm = cm_from_img_tuple(item, class_names, iou_threshold)
        cm_copy = cm.copy()
        cm[:, len(class_names)] = cm_copy[len(class_names), :]
        cm[len(class_names), :] = cm_copy[:, len(class_names)]
        cms.append(cm)
        if save_individual_images:
            cmname = save_image(
                item,
                test_imgs_dir,
                class_names,
                iou_threshold,
                conf_threshold,
                cm,
            )
            save_metrics(
                cm,
                class_names,
                op.join(
                    test_imgs_dir,
                    f"{cmname}_{iou_threshold}_{conf_threshold}_metrics.csv",
                ),
                iou_threshold=iou_threshold,
            )
    print("Images saved with bounding boxes drawn.")
    bigcm = np.sum(np.array(cms), axis=0)
    cnames_bground = class_names.copy().append("background")
    pd.DataFrame(bigcm, index=cnames_bground, columns=cnames_bground).to_csv(
        op.join(outdir, f"{model_type}_confusion_matrix.csv")
    )
    plot_title = model_type.replace("_", " ").title()
    print(class_names)
    plot_cm_sn(
        bigcm,
        class_names,
        outpath=op.join(
            outdir,
            f"{model_type}_{iou_threshold}_{conf_threshold}_confusion_matrix_predicted_normed.png",
        ),
        title=f"{plot_title} Confusion Matrix, Proportions",
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
        normalize=True,
        norm_axis="predicted",
    )
    plot_cm_sn(
        bigcm,
        class_names,
        outpath=op.join(
            outdir,
            f"{model_type}_{iou_threshold}_{conf_threshold}_confusion_matrix_true_normed.png",
        ),
        title=f"{plot_title} Confusion Matrix, Proportions",
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
        normalize=True,
        norm_axis="true",
    )
    plot_cm_sn(
        bigcm,
        class_names,
        outpath=op.join(
            outdir,
            f"{model_type}_{iou_threshold}_{conf_threshold}_confusion_matrix_counts.png",
        ),
        title=f"{plot_title} Confusion Matrix, Counts",
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
        normalize=False,
        cmap="Greens",
    )
    save_metrics(
        bigcm,
        class_names,
        op.join(
            outdir,
            f"{model_type}_{iou_threshold}_{conf_threshold}_metrics.csv",
        ),
        iou_threshold=iou_threshold,
    )
    class_str = "\n".join(class_names)
    print(
        f"Complete, Confusion matrix, images, and metrics saved in {outdir} for these classes: {class_str}"
    )


if __name__ == "__main__":
    run_eval()
