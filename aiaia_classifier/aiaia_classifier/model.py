#!/usr/bin/env python
"""
Script specifying TF estimator object for training/serving
@author: DevelopmentSeed
"""

import glob
import json
import os
import os.path
from functools import partial
from typing import Iterator

import numpy as np
import pandas as pd
import tensorflow as tf
from absl import app, flags, logging
from sklearn.metrics import fbeta_score, precision_score, recall_score
from tensorflow.keras.applications import Xception
from tensorflow.keras.estimator import model_to_estimator
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tqdm import tqdm
from utils_readtfrecords import (
    get_dataset_feeder,
    country_file_patterns,
    parse_and_augment_fn,
    parse_fn,
)
from utils_train import FBetaScore, upload_file_to_gcs

FLAGS = flags.FLAGS

# Override these flags when calling model.py or in the katib yaml files if
# running experiments via katib
flags.DEFINE_integer("n_classes", 2, "Number of classes in dataset")
flags.DEFINE_list("class_names", ["not_object", "object"], "class names in data set")
flags.DEFINE_list("countries", [""], "countries to use for input")
flags.DEFINE_integer("img_dim", 400, "dimensions of img array")
flags.DEFINE_integer("num_channels", 3, "number of channels in img array")
flags.DEFINE_string("x_feature_name", "input_1", "layer name")
flags.DEFINE_integer("n_map_threads", 4, "threads")
flags.DEFINE_integer("shuffle_buffer_size", None, "should match size of train dataset")
flags.DEFINE_integer("prefetch_buffer_size", 1, "prefetch buffer size")
flags.DEFINE_string(
    "tf_train_data_dir", None, "Path to training data dir. Relative to local_dataset_dir."
)
flags.DEFINE_string(
    "tf_val_data_dir", None, "Path to val data dir. Relative to local_dataset_dir."

flags.DEFINE_string(
    "model_logs_dir",
    "model_logs",
    "Directory to write evaluation, tf records, checkpoints after training",
)

flags.DEFINE_string(
    "local_dataset_dir", None, "local directory to copy data into."
)
flags.DEFINE_string(
    "model_output_dir", "model_output", "Path to save models."
)
flags.DEFINE_string(
    "results_dir",
    None,
    "Path to GCS to write results",
)
flags.DEFINE_string("model_id", "s", "model id for saving.")
flags.DEFINE_string(
    "model_upload_id",
    None,
    "model id for upload paths that will be unique to each",
)
flags.DEFINE_string(
    "tf_test_ckpt_path",
    None,
    "Use to override training and run prediction on test data.",
)
flags.DEFINE_integer(
    "tf_steps_per_summary", 10, "Training steps per Tensorboard events save."
)
flags.DEFINE_integer(
    "tf_steps_per_checkpoint", 100, "Training steps per model checkpoint save."
)
flags.DEFINE_integer("tf_batch_size", 16, "Size of one batch for training")
flags.DEFINE_integer("tf_train_steps", 200, "The number of training steps to perform")
flags.DEFINE_integer("tf_dense_size_a", 256, "Size of final dense hidden layer")
flags.DEFINE_float(
    "tf_dense_dropout_rate_a", 0.3, "Dropout rate of the final dense hidden layer"
)
flags.DEFINE_integer("tf_dense_size", 128, "Size of final dense hidden layer")
flags.DEFINE_float(
    "tf_dense_dropout_rate", 0.35, "Dropout rate of the final dense hidden layer"
)
flags.DEFINE_string("tf_dense_activation", "relu", "Activation output layer")
flags.DEFINE_float("tf_learning_rate", 0.001, "learning rate for training")
flags.DEFINE_enum(
    "tf_optimizer", "adam", ["adam", "sgd", "rmsprop"], "Optimizer function"
)


######################
# Modeling Code
######################
def resnet50_estimator(params, model_dir, run_config):
    """Get a Resnet50 model as a tf.estimator object"""

    # Get the original resnet model pre-initialized weights
    base_model = Xception(
        weights="imagenet",
        include_top=False,  # Peel off top layer
        pooling="avg",
        input_shape=[params["img_dim"], params["img_dim"], params["num_channels"]],
    )
    # Get final layer of base Resnet50 model
    x = base_model.output
    # Add a fully-connected layer
    x = Dense(
        params["dense_size_a"],
        activation=params["dense_activation"],
        name="dense",
    )(x)
    # Add (optional) dropout and output layer
    x = Dropout(rate=params["dense_dropout_rate_a"])(x)
    x = Dense(
        params["dense_size"],
        activation=params["dense_activation"],
        name="dense_preoutput",
    )(x)
    x = Dropout(rate=params["dense_dropout_rate"])(x)
    output = Dense(params["n_classes"], name="output", activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    # Get (potentially decaying) learning rate
    optimizer = create_optimizer(params["optimizer"], params["learning_rate"])
    model.compile(optimizer=optimizer, loss=params["loss"], metrics=params["metrics"])

    return model_to_estimator(
        keras_model=model,
        model_dir=os.path.join(model_dir, FLAGS.model_id),
        config=run_config,
    )


def create_optimizer(name, learning_rate, momentum=0.9):
    """Helper to get optimizer from text params"""
    if name == "adam":
        return Adam(learning_rate=learning_rate)
    if name == "sgd":
        return SGD(learning_rate=learning_rate)
    if name == "rmsprop":
        return RMSprop(learning_rate=learning_rate, momentum=momentum)
    raise ValueError(f"Unsupported optimizer: {name}")


def resnet_serving_input_receiver_fn():
    """Convert b64 string encoded images into a tensor for production"""

    def decode_and_resize(image_str_tensor):
        """Decodes image string, resizes it and returns a uint8 tensor."""
        image = tf.image.decode_image(image_str_tensor, channels=3, dtype=tf.uint8)
        image = tf.reshape(image, [FLAGS.img_dim, FLAGS.img_dim, FLAGS.num_channels])
        return image

    # Run processing for batch prediction.
    input_ph = tf.compat.v1.placeholder(tf.string, shape=[None], name="image_binary")

    with tf.device("/cpu:0"):
        images_tensor = tf.map_fn(
            decode_and_resize, input_ph, back_prop=False, dtype=tf.uint8
        )

    # Cast to float
    images_tensor = tf.cast(images_tensor, dtype=tf.float32)

    # re-scale pixel values between 0 and 1
    images_tensor = tf.divide(images_tensor, 255)

    return tf.estimator.export.ServingInputReceiver(
        {FLAGS.x_feature_name: images_tensor}, {"image_bytes": input_ph}
    )


def rebase(src_base: str, dest_base: str, src_path: str) -> str:
    """Rebase a file path from one base path to another base path.

    For example, given a source path of /dir/subdir/path/to/file, a source base of
    /dir/subdir (the source base must be a prefix, or subpath, of the source path),
    and a destination base of /dest/base, the rebased path would be
    /dest/base/path/to/file.

    Note that leading slashes are not required for any of the arguments, only that the
    source base is a prefix of the source path.  If the source base is not a prefix of
    the source path, the result is unpredictable.  When the source base is identical to
    the source path, the result is the destination base with a trailing path separator
    and dot ('.').

    >>> rebase("/absolute/src", "/dest/base", "/absolute/src/path/to/file")
    '/dest/base/path/to/file'
    >>> rebase("/absolute/src", "/dest/base", "/absolute/src")
    '/dest/base/.'
    >>> rebase("relative/src", "/dest/base", "relative/src/path/to/file")
    '/dest/base/path/to/file'
    >>> rebase("relative/src", "base/relative", "relative/src/path/to/file")
    'base/relative/path/to/file'
    """
    return os.path.join(dest_base, os.path.relpath(src_path, src_base))


def ifilepaths(dirpath: str, *, recursive=True) -> Iterator[str]:
    """Return an iterator that yields the paths of all regular files in a directory.

    By default, directories are traversed recursively, but regardless, only "regular"
    file paths are yielded, not directory paths.
    """
    ipaths = glob.iglob(os.path.join(dirpath, "**"), recursive=recursive)
    return filter(os.path.isfile, ipaths)


def upload_files(bucket_name: str, blob_name_base: str, src_dir: str) -> None:
    """Upload all files (recursively) in a directory to a GCS bucket.

    All files within the directory are uploaded relative to the given base blob name,
    such that their paths relative to the given directory are made relative to the base
    blob name.  For example, a file at <src_dir>/relpath/to/file is uploaded to a blob
    named <blob_name_base>/relpath/to/file.
    """
    for filepath in ifilepaths(src_dir):
        blob_name = rebase(src_dir, blob_name_base, filepath)
        upload_file_to_gcs(bucket_name, blob_name, filepath)


def main(_):
    """
    Function to run TF Estimator
    Note: set the `TF_CONFIG` environment variable according to:
    https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate
    if running locally. Otherwise, AzureML sets the TF_CONFIG automatically: 
    https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-tensorflow
    """

    run_config = tf.estimator.RunConfig(
        model_dir=os.path.join(FLAGS.tf_model_dir, FLAGS.model_id),
        save_summary_steps=FLAGS.tf_steps_per_summary,
        save_checkpoints_steps=FLAGS.tf_steps_per_checkpoint,
        log_step_count_steps=FLAGS.tf_steps_per_summary,
    )

    model_params = {
        "n_classes": FLAGS.n_classes,
        "img_dim": FLAGS.img_dim,
        "num_channels": FLAGS.num_channels,
        "train_steps": FLAGS.tf_train_steps,
        "dense_size_a": FLAGS.tf_dense_size_a,
        "dense_size": FLAGS.tf_dense_size,
        "dense_activation": FLAGS.tf_dense_activation,
        "dense_dropout_rate_a": FLAGS.tf_dense_dropout_rate_a,
        "dense_dropout_rate": FLAGS.tf_dense_dropout_rate,
        "optimizer": FLAGS.tf_optimizer,
        "metrics": [
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            FBetaScore(FLAGS.n_classes, beta=2.0, average="weighted"),
        ],
        "learning_rate": FLAGS.tf_learning_rate,
        # "loss": sigmoid_focal_crossentropy,
        "loss": tf.keras.losses.BinaryCrossentropy(),
    }

    def precision_m(labels, predictions):
        precision_metric = tf.keras.metrics.Precision(name="precision_m")
        precision_metric.update_state(y_true=labels, y_pred=predictions["output"])
        return {"precision_m": precision_metric}

    def recall_m(labels, predictions):
        recall_metric = tf.keras.metrics.Recall(name="recall_m")
        recall_metric.update_state(y_true=labels, y_pred=predictions["output"])
        return {"recall_m": recall_metric}

    def fbeta_m(labels, predictions):
        fbeta_metric = FBetaScore(
            FLAGS.n_classes, beta=2.0, average="weighted", threshold=0.5
        )
        fbeta_metric.update_state(y_true=labels, y_pred=predictions["output"])
        return {"fbeta_m": fbeta_metric}

    classifier = resnet50_estimator(model_params, FLAGS.tf_model_dir, run_config)
    classifier = tf.estimator.add_metrics(classifier, fbeta_m)
    classifier = tf.estimator.add_metrics(classifier, precision_m)
    classifier = tf.estimator.add_metrics(classifier, recall_m)

    ###################################
    # Check if user wants to run test
    ###################################
    # Create test dataset function if needed
    if FLAGS.tf_test_ckpt_path:
        print(
            "Overriding training and running test set prediction with model ckpt:",
            FLAGS.tf_test_ckpt_path,
        )

        os.makedirs(FLAGS.tf_test_results_dir, exist_ok=True)

        logging.info("Beginning test for model")

        # Create test data function, get `y_true`
        test_file_patterns = country_file_patterns(
            FLAGS.tf_test_data_dir,
            "test*{country}*.tfrecords",
            FLAGS.countries,
        )
        logging.info(test_file_patterns)
        map_func = partial(parse_fn, n_chan=3, n_classes=model_params["n_classes"])

        dataset_test = get_dataset_feeder(
            file_patterns=test_file_patterns,
            data_map_func=map_func,
            repeat=False,
            n_map_threads=FLAGS.n_map_threads,
            batch_size=1,  # Use bs=1 here to count samples instead of batches
            prefetch_buffer_size=1,
        )

        y_true = [features[1].numpy()[0] for features in dataset_test]
        print(f"Found {len(y_true)} total samples to test.")

        # Reset the dataset iteration for prediction
        dataset_test_fn = partial(
            get_dataset_feeder,
            file_patterns=test_file_patterns,
            data_map_func=map_func,
            repeat=False,
            n_map_threads=FLAGS.n_map_threads,
            batch_size=FLAGS.tf_batch_size,
            prefetch_buffer_size=1,
        )

        # Reset test iterator and run predictions
        raw_preds = classifier.predict(
            dataset_test_fn,
            yield_single_examples=True,
            checkpoint_path=FLAGS.tf_test_ckpt_path,
        )

        p_list = [raw_pred["output"] for raw_pred in raw_preds]
        preds = [(p >= 0.5).astype(int) for p in tqdm(p_list, miniters=1000)]

        output_d = {
            "raw_prediction": p_list,
            "threshold": preds,
            "true-label": y_true,
        }

        df_pred = pd.DataFrame.from_dict(output_d)
        df_pred.to_csv(os.path.join(FLAGS.tf_test_results_dir, "preds.csv"))

        print("preds csv written")

        recall_scores = [
            recall_score(np.array(y_true)[:, i], np.array(preds)[:, i])
            for i in np.arange(0, FLAGS.n_classes)
        ]

        precision_scores = [
            precision_score(np.array(y_true)[:, i], np.array(preds)[:, i])
            for i in np.arange(0, FLAGS.n_classes)
        ]

        fbeta_scores = [
            fbeta_score(np.array(y_true)[:, i], np.array(preds)[:, i], beta=2)
            for i in np.arange(0, FLAGS.n_classes)
        ]

        d = {
            "Precision": precision_scores,
            "Recall": recall_scores,
            "fbeta_2": fbeta_scores,
            "POI": model_params["class_names"],
        }

        df = pd.DataFrame.from_dict(d)
        df["POI"] = model_params["class_names"]
        df.to_csv(os.path.join(FLAGS.tf_test_results_dir, "test_stats.csv"))

        print("test stats written")
        return d

    ###################################
    # Create data feeder functions
    ###################################

    # download data from GCS

    # Create training dataset function

    train_file_patterns = country_file_patterns(
        FLAGS.tf_train_data_dir,
        "train*{country}*.tfrecords",
        FLAGS.countries,
    )

    map_func = partial(
        parse_and_augment_fn,
        n_chan=3,
        n_classes=model_params["n_classes"],
        img_dim=model_params["img_dim"],
    )

    dataset_train_fn = partial(
        get_dataset_feeder,
        file_patterns=train_file_patterns,
        data_map_func=map_func,
        repeat=True,
        n_map_threads=FLAGS.n_map_threads,
        batch_size=FLAGS.tf_batch_size,
        prefetch_buffer_size=FLAGS.prefetch_buffer_size,
    )

    # Create validation dataset function
    val_file_patterns = country_file_patterns(
        FLAGS.tf_val_data_dir,
        "val*{country}*.tfrecords",
        FLAGS.countries,
    )

    map_func = partial(
        parse_and_augment_fn,
        n_chan=3,
        n_classes=model_params["n_classes"],
        img_dim=model_params["img_dim"],
    )

    dataset_validate_fn = partial(
        get_dataset_feeder,
        file_patterns=val_file_patterns,
        data_map_func=map_func,
        repeat=True,
        n_map_threads=FLAGS.n_map_threads,
        batch_size=FLAGS.tf_batch_size,
        prefetch_buffer_size=FLAGS.prefetch_buffer_size,
    )

    ###################################
    # Run train/val w/ estimator object
    ###################################

    # Set up train and evaluation specifications
    train_spec = tf.estimator.TrainSpec(
        input_fn=dataset_train_fn, max_steps=FLAGS.tf_train_steps
    )

    logging.info("export final pre")
    export_final = tf.estimator.FinalExporter(
        FLAGS.model_id,
        serving_input_receiver_fn=resnet_serving_input_receiver_fn,
    )
    logging.info("export final post")

    # Not sure why we need to count the number of val samples and use the count as the
    # number of steps for the following EvalSpec.  Perhaps Martha knows.
    val_files = tf.io.matching_files(val_file_patterns)
    n_val_samples = sum(1 for _ in tf.data.TFRecordDataset(val_files))

    eval_spec = tf.estimator.EvalSpec(
        input_fn=dataset_validate_fn,
        steps=n_val_samples,
        exporters=export_final,
        throttle_secs=1,
        start_delay_secs=1,
    )
    logging.info("eval spec")

    ###################################
    # Run training, save if needed
    ###################################
    logging.info("train and evaluate")
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    logging.info("training done.")

if __name__ == "__main__":
    app.run(main)
