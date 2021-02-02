import pytest
import sys
import os
from os import path as op
import tensorflow as tf  # version 2
import tensorflow.compat.v1 as tf1
import numpy as np
import glob

# path to aiaia_detector module
module_folder = op.dirname(op.dirname(op.abspath(__file__)))
sys.path.append(module_folder)
from evaluation import *

os.chdir(op.dirname(op.abspath(__file__)))


@pytest.fixture
def tfrecords_folder():
    return "./data"


@pytest.fixture
def model_rel_path():
    return "./frozen_inference_graph.pb"

@pytest.fixture
def class_names():
    return [
        "buffalo",
        "dark_coloured_large",
        "elephant",
        "giraffe",
        "hippopotamus",
        "light_coloured_large",
        "smaller_ungulates",
        "warthog",
        "zebra",
        "background",
    ]


def test_read_all_tfrecords(tfrecords_folder):
    # Reading all images and groundtruth
    output_dir = "./outputs"
    all_gt = []
    all_imgs_and_gt = []
    # list tfrecord folders
    print(op.abspath(tfrecords_folder))
    tf_record_paths = glob.glob(tfrecords_folder + "/*")
    assert len(tf_record_paths) == 1
    for tfrecord_path in glob.glob(tfrecords_folder + "/*"):
        assert tfrecord_path.endswith(".tfrecords")
        sub_dir = op.join(output_dir, "")
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        items = get_example(tfrecord_path)
        items_no_imgs = get_example_no_img(tfrecord_path)
        all_gt.extend(items_no_imgs)
        all_imgs_and_gt.extend(items)
    assert len(all_imgs_and_gt) == 20
    import shutil

    shutil.rmtree(output_dir)


# TODO, what's a more concise way to test linear processing workflows with pytest?
# How can state be shared from one test to another?
def test_run_inference_and_filter(tfrecords_folder, model_rel_path):
    output_dir = "./outputs"
    # Reading all images and groundtruth
    all_gt = []
    all_imgs_and_gt = []
    for tfrecord_path in glob.glob(tfrecords_folder + "/*"):
        sub_dir = op.join(output_dir, "")
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        items = get_example(tfrecord_path)
        items_no_imgs = get_example_no_img(tfrecord_path)
        all_gt.extend(items_no_imgs)
        all_imgs_and_gt.extend(items)
    # Testing the tf model on subset of test set, all_imgs_and_gt
    model_path = op.join(os.getcwd(), model_rel_path)
    assert os.path.exists(model_path)
    detection_graph = tf1.Graph()
    with detection_graph.as_default():
        od_graph_def = tf1.GraphDef()
        with tf1.gfile.GFile(model_path, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf1.import_graph_def(od_graph_def, name="")
    det_dicts = tf1_od_pred(all_imgs_and_gt, detection_graph)
    (
        all_gt,
        det_dicts_filtered,
        all_gt_boxes,
        all_detection_boxes,
        all_gt_classes,
        all_det_classes,
    ) = filter_detections_and_concat_boxes(all_gt, det_dicts, conf_threshold=0.5)
    assert len(det_dicts_filtered) == 20
    import shutil

    shutil.rmtree(output_dir)


def test_save_output_img(tfrecords_folder, model_rel_path, class_names):
    output_dir = "./outputs"
    # Reading all images and groundtruth
    all_gt = []
    all_imgs_and_gt = []
    for tfrecord_path in glob.glob(tfrecords_folder + "/*"):
        sub_dir = op.join(output_dir, "")
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        items = get_example(tfrecord_path)
        items_no_imgs = get_example_no_img(tfrecord_path)
        all_gt.extend(items_no_imgs)
        all_imgs_and_gt.extend(items)
    # Testing the tf model on subset of test set, all_imgs_and_gt
    model_path = op.join(os.getcwd(), model_rel_path)
    assert os.path.exists(model_path)
    detection_graph = tf1.Graph()
    with detection_graph.as_default():
        od_graph_def = tf1.GraphDef()
        with tf1.gfile.GFile(model_path, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf1.import_graph_def(od_graph_def, name="")
    det_dicts = tf1_od_pred(all_imgs_and_gt, detection_graph)
    (
        all_gt,
        det_dicts_filtered,
        all_gt_boxes,
        all_detection_boxes,
        all_gt_classes,
        all_det_classes,
    ) = filter_detections_and_concat_boxes(all_gt, det_dicts, conf_threshold=0.5)
    # save out images with both boxes drawn for sanity check
    test_imgs_dir = os.path.join(output_dir, "test_imgs_with_boxes")
    all_imgs_gt_det = list(zip(all_imgs_and_gt, det_dicts_filtered))
    if not os.path.exists(test_imgs_dir):
        os.makedirs(test_imgs_dir)

    cms = []
    for item in all_imgs_gt_det:
        cm = cm_from_img_tuple(item)
        cms.append(cm)
        save_image(item, test_imgs_dir, class_names, cm)
        assert cm.shape == (10, 10)

    import shutil

    shutil.rmtree(output_dir)


def test_plot_overall_cm(tfrecords_folder, model_rel_path, class_names):
    output_dir = "./outputs"
    # Reading all images and groundtruth
    all_gt = []
    all_imgs_and_gt = []
    for tfrecord_path in glob.glob(tfrecords_folder + "/*"):
        sub_dir = op.join(output_dir, "")
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        items = get_example(tfrecord_path)
        items_no_imgs = get_example_no_img(tfrecord_path)
        all_gt.extend(items_no_imgs)
        all_imgs_and_gt.extend(items)
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
    (
        all_gt,
        det_dicts_filtered,
        all_gt_boxes,
        all_detection_boxes,
        all_gt_classes,
        all_det_classes,
    ) = filter_detections_and_concat_boxes(all_gt, det_dicts, conf_threshold=0.5)
    # save out images with both boxes drawn for sanity check
    test_imgs_dir = os.path.join(output_dir, "test_imgs_with_boxes")
    all_imgs_gt_det = list(zip(all_imgs_and_gt, det_dicts_filtered))
    if not os.path.exists(test_imgs_dir):
        os.makedirs(test_imgs_dir)

    cms = []
    for item in all_imgs_gt_det:
        cm = cm_from_img_tuple(item)
        cms.append(cm)
        save_image(item, test_imgs_dir, class_names, cm)
    bigcm = np.sum(np.array(cms), axis=0)
    bigcm_path = os.path.join(output_dir, "./cm.png")
    plot_confusion_matrix(bigcm, class_names, outpath=bigcm_path)
    assert os.path.exists(bigcm_path)
    import shutil

    shutil.rmtree(output_dir)
