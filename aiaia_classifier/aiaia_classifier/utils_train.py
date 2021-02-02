"""
Utility functions for the training process
@author: DevelopmentSeed
"""
from __future__ import absolute_import, division, print_function

import os
import os.path as op
import shutil
import time
import zipfile

import tensorflow as tf
from absl import logging
from google.cloud import storage
from sklearn.metrics import fbeta_score
from tenacity import retry, stop_after_attempt, wait_exponential


# GCP Helper Functions
def check_create_dir(dir_path):
    """Create a directory if it does not exist."""
    if not op.isdir(dir_path):
        os.mkdir(dir_path)


def print_start_details(start_time):
    """Print config at the start of a run."""
    print("Start time: " + start_time.strftime("%d/%m %H:%M:%S"))
    print("\n\n" + "=" * 40)


def print_end_details(start_time, end_time):
    """Print runtime information."""
    run_time = end_time - start_time
    hours, remainder = divmod(run_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    print("\n\n" + "=" * 40)
    print("End time: " + end_time.strftime("%d/%m %H:%M:%S"))
    print("Total runtime: %i:%02i:%02i" % (hours, minutes, seconds))


def copy_filenames_to_dir(file_list, dst_dir):
    """Copy a list of filenames (like images) to new directory."""
    for file_name in file_list:
        print("Copying: %s to %s" % (file_name, dst_dir))
        shutil.copy(file_name, dst_dir)
    print("Done.")


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(10))
def upload_file_to_gcs(
    bucket_name,
    blob_name,
    filename,
    chunk_size=(256 * 1024) * 256,
):
    """Uploads a file to a google cloud storage bucket.
    Parameters
    ----------
    filename: str
        Name of the file on disk.
    bucket_name: str
        Name of the GCS bucket to upload to.
    blob_name: str
        Name of the blob on GCS
    chuck_size: int
        Size of each chunk in bytes. Must be multiple of 256kB
        (256*1024 bytes). Setting this helps large files upload properly.
        Default 256*1024*256, which is 64MB.
    """

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name, chunk_size)
    blob.upload_from_filename(filename)


def list_and_download_blobs(bucket_name, destination_file_dir):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)

    for blob in blobs:
        destination_file_name = destination_file_dir + "/" + blob.name
        download_blob(bucket_name, blob.name, destination_file_name)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))


def download_tfr(
    bucket_name, source_blob_name, destination_file_name, destination_file_dir
):
    download_blob(bucket_name, source_blob_name, destination_file_name)
    with zipfile.ZipFile(destination_file_name, "r") as z:
        z.extractall(destination_file_dir)


def download_gcs_blobs_with_prefix(
    bucket_name, prefix, destination_dir, delimiter=None, clobber=False, retries=10
):
    """Downloads all blobs from a GCS bucket that begin with the prefix."""

    storage_client = storage.Client()

    logging.info("Downloading from {bucket_name} to {destination_dir}")
    check_create_dir(destination_dir)

    while True:
        try:
            blobs = storage_client.list_blobs(
                bucket_name, prefix=prefix, delimiter=delimiter
            )

            for blob in blobs:
                fname = blob.name.split("/")[-1]
                fpath = op.join(destination_dir, fname)
                if not op.exists(fpath) or clobber:
                    blob.download_to_filename(fpath, client=storage_client)
                else:
                    logging.info("exists, skipping")

                if op.splitext(fname)[-1] == ".zip":
                    logging.info("Extracting {fpath}.")
                    with zipfile.ZipFile(fpath, "r") as z:
                        z.extractall(destination_dir)
            break
        except Exception as e:
            retries -= 1
            logging.info("Error during download.")

            if retries:
                time.sleep(10)
                continue
            else:
                raise  # Reraise the last exception


def download_training_data(bucket_name, prefixes, destination_dirs):
    """Download a directory on GCS to target local directory."""

    for prefix, destination_dir in zip(prefixes, destination_dirs):
        download_gcs_blobs_with_prefix(bucket_name, prefix, destination_dir)


def fbeta(true_label, prediction):
    return fbeta_score(true_label, prediction, beta=1, average="samples")


class FBetaScore(tf.keras.metrics.Metric):
    """Computes F-Beta score.
    It is the weighted harmonic mean of precision
    and recall. Output range is [0, 1]. Works for
    both multi-class and multi-label classification.
    F-Beta = (1 + beta^2) * (prec * recall) / ((beta^2 * prec) + recall)
    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro` and
            `weighted`. Default value is None.
        beta: Determines the weight of precision and recall
            in harmonic mean. Determines the weight given to the
            precision and recall. Default value is 1.
        threshold: Elements of `y_pred` greater than threshold are
            converted to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
    Returns:
        F-Beta Score: float
    Raises:
        ValueError: If the `average` has values other than
        [None, micro, macro, weighted].
        ValueError: If the `beta` value is less than or equal
        to 0.
    `average` parameter behavior:
        None: Scores for each class are returned
        micro: True positivies, false positives and
            false negatives are computed globally.
        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.
        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.
    """

    def __init__(
        self,
        num_classes,
        average=None,
        beta=1.0,
        threshold=None,
        name="fbeta_score",
        dtype=tf.float32,
    ):
        super(FBetaScore, self).__init__(name=name)

        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError(
                "Unknown average type. Acceptable values "
                "are: [None, micro, macro, weighted]"
            )

        if not isinstance(beta, float):
            raise TypeError("The value of beta should be a python float")

        if beta <= 0.0:
            raise ValueError("beta value should be greater than zero")

        if threshold is not None:
            if not isinstance(threshold, float):
                raise TypeError("The value of threshold should be a python float")
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError("threshold should be between 0 and 1")

        self.num_classes = num_classes
        self.average = average
        self.beta = beta
        self.threshold = threshold
        self.axis = None
        self.init_shape = []

        if self.average != "micro":
            self.axis = 0
            self.init_shape = [self.num_classes]

        def _zero_wt_init(name):
            return self.add_weight(
                name, shape=self.init_shape, initializer="zeros", dtype=self.dtype
            )

        self.true_positives = _zero_wt_init("true_positives")
        self.false_positives = _zero_wt_init("false_positives")
        self.false_negatives = _zero_wt_init("false_negatives")
        self.weights_intermediate = _zero_wt_init("weights_intermediate")

    # TODO: Add sample_weight support, currently it is
    # ignored during calculations.
    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.threshold is None:
            threshold = tf.reduce_max(y_pred, axis=-1, keepdims=True)
            # make sure [0, 0, 0] doesn't become [1, 1, 1]
            # Use abs(x) > eps, instead of x != 0 to check for zero
            y_pred = tf.logical_and(y_pred >= threshold, tf.abs(y_pred) > 1e-12)
        else:
            y_pred = y_pred > self.threshold

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        def _count_non_zero(val):
            non_zeros = tf.math.count_nonzero(val, axis=self.axis)
            return tf.cast(non_zeros, self.dtype)

        self.true_positives.assign_add(_count_non_zero(y_pred * y_true))
        self.false_positives.assign_add(_count_non_zero(y_pred * (y_true - 1)))
        self.false_negatives.assign_add(_count_non_zero((y_pred - 1) * y_true))
        self.weights_intermediate.assign_add(_count_non_zero(y_true))

    def result(self):
        precision = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_positives
        )
        recall = tf.math.divide_no_nan(
            self.true_positives, self.true_positives + self.false_negatives
        )

        mul_value = precision * recall
        add_value = (tf.math.square(self.beta) * precision) + recall
        mean = tf.math.divide_no_nan(mul_value, add_value)
        f1_score = mean * (1 + tf.math.square(self.beta))

        if self.average == "weighted":
            weights = tf.math.divide_no_nan(
                self.weights_intermediate, tf.reduce_sum(self.weights_intermediate)
            )
            f1_score = tf.reduce_sum(f1_score * weights)

        elif self.average is not None:  # [micro, macro]
            f1_score = tf.reduce_mean(f1_score)

        return f1_score

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "average": self.average,
            "beta": self.beta,
        }

        if self.threshold is not None:
            config["threshold"] = self.threshold

        base_config = super(FBetaScore, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        self.true_positives.assign(tf.zeros(self.init_shape, self.dtype))
        self.false_positives.assign(tf.zeros(self.init_shape, self.dtype))
        self.false_negatives.assign(tf.zeros(self.init_shape, self.dtype))
        self.weights_intermediate.assign(tf.zeros(self.init_shape, self.dtype))


class F1Score(FBetaScore):
    """Computes F-1 Score.
    It is the harmonic mean of precision and recall.
    Output range is [0, 1]. Works for both multi-class
    and multi-label classification.
    F-1 = 2 * (precision * recall) / (precision + recall)
    Args:
        num_classes: Number of unique classes in the dataset.
        average: Type of averaging to be performed on data.
            Acceptable values are `None`, `micro`, `macro`
            and `weighted`. Default value is None.
        threshold: Elements of `y_pred` above threshold are
            considered to be 1, and the rest 0. If threshold is
            None, the argmax is converted to 1, and the rest 0.
    Returns:
        F-1 Score: float
    Raises:
        ValueError: If the `average` has values other than
        [None, micro, macro, weighted].
    `average` parameter behavior:
        None: Scores for each class are returned
        micro: True positivies, false positives and
            false negatives are computed globally.
        macro: True positivies, false positives and
            false negatives are computed for each class
            and their unweighted mean is returned.
        weighted: Metrics are computed for each class
            and returns the mean weighted by the
            number of true instances in each class.
    """

    def __init__(
        self,
        num_classes,
        average=None,
        threshold=None,
        name="f1_score",
        dtype=tf.float32,
    ):
        super(F1Score, self).__init__(
            num_classes, average, 1.0, threshold, name=name, dtype=dtype
        )

    def get_config(self):
        base_config = super(F1Score, self).get_config()
        del base_config["beta"]
        return base_config
