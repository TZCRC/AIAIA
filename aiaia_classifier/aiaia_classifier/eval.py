"""
Model evaluation script for aiaia classifier.
Author: @DevelopmentSeed

Run locally:
python3 aiaia_classifier/eval.py \
       -tf_test_data_dir=Hoduras_test_data/ \
       --tf_test_ckpt_path=hondorus_model_ckpt/country_models_model_outputs_Honduras_v1_model.ckpt-6000 \
       --tf_test_results_dir=model_eval

Reading files from GCS:

python3 aiaia_classifier/eval.py \
        --tf_test_data_dir='gs://regional_model/country_models/Honduras/' \
        --tf_test_ckpt_path='gs://regional_model/country_models/model_outputs/Honduras/v1/model.ckpt-6000' \
        --tf_test_results_dir=model_eval
"""

import os
import glob
from os import makedirs, path as op
import matplotlib

matplotlib.use("agg")
# import tkinter
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import tensorflow as tf
from functools import partial
from absl import app, flags, logging
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from scipy.spatial.distance import euclidean
from model import resnet50_estimator
from tqdm import tqdm
from utils_readtfrecords import (
    get_dataset_feeder,
    country_file_patterns,
    parse_fn,
)
from utils_train import FBetaScore

# import matplotlib as mpl
FLAGS = flags.FLAGS


def plot_roc(y_test_arr, y_pred_arr, plot_dir, countries):
    """
    Plot ROC curve
    """
    if not op.isdir(plot_dir):
        makedirs(plot_dir)
    plot_dir = plot_dir

    y_neg_pred = y_pred_arr[
        y_test_arr == 0
    ]  # Predictions for negative examples
    y_pos_pred = y_pred_arr[
        y_test_arr == 1
    ]  # Predictions for positive examples

    # Accuracy (should match tensorboard)
    correct = np.sum(y_test_arr == np.round(y_pred_arr))
    total = y_test_arr.shape[0]
    acc = float(correct) / float(total)
    print("Accuracy: {:0.5f}".format(acc))
    # Compute FPR, TPR for '1' label (i.e., positive examples)
    fpr, tpr, thresh = roc_curve(y_test_arr, y_pred_arr)
    roc_auc = auc(fpr, tpr)

    # Min corner dist (*one* optimal value for threshold derived from ROC curve)
    corner_dists = np.empty((fpr.shape[0]))
    for di, (x_val, y_val) in enumerate(zip(fpr, tpr)):
        corner_dists[di] = euclidean([0.0, 1.0], [x_val, y_val])
    opt_cutoff_ind = np.argmin(corner_dists)
    min_corner_x = fpr[opt_cutoff_ind]
    min_corner_y = tpr[opt_cutoff_ind]

    ####################
    # Plot
    ####################
    print("Plotting.")
    # plt.close('all')
    sns.set()
    sns.set_style("darkgrid", {"axes.facecolor": ".9"})
    sns.set_context("talk", font_scale=1.1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label="ROC curve (area={:0.2f})".format(roc_auc))
    ax.plot(
        [min_corner_x, min_corner_x],
        [0, min_corner_y],
        color="r",
        lw=1,
        label="Min-corner distance\n(FPR={:0.2f}, thresh={:0.2f})".format(
            min_corner_x, thresh[opt_cutoff_ind]
        ),
    )
    plt.plot([0, 1], [0, 1], color="black", lw=0.75, linestyle="--")
    ax.set_xlim([-0.03, 1.0])
    ax.set_ylim([0.0, 1.03])
    ax.set_xlabel("False Positive Rate\n(1 - Specificity)")
    ax.set_ylabel("True Positive Rate\n(Sensitivity)")
    ax.set_aspect("equal")
    ax.set_title("ROC curve for aiaia binary classification")
    plt.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(op.join(plot_dir, f"roc_{countries[0]}.png"))

    # Plot a kernel density estimate and rug plot
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))
    kde_kws = dict(shade=True, clip=[0.0, 1.0], alpha=0.3)
    rug_kws = dict(alpha=0.2)
    sns.distplot(
        y_neg_pred,
        hist=False,
        kde=True,
        rug=True,
        norm_hist=True,
        color="b",
        kde_kws=kde_kws,
        rug_kws=rug_kws,
        label="True negatives",
        ax=ax2,
    )
    sns.distplot(
        y_pos_pred,
        hist=False,
        kde=True,
        rug=True,
        norm_hist=True,
        color="r",
        kde_kws=kde_kws,
        rug_kws=rug_kws,
        label="True positives",
        ax=ax2,
    )
    ax2.set_title("Predicted scores for true positives and true negatives")
    ax2.set_xlim([0.0, 1.0])
    ax2.set_xlabel("Model's predicted score")
    ax2.set_ylabel("Probability density")
    plt.legend(loc="best")
    fig2.savefig(op.join(plot_dir, f"dist_fpr_tpr_{countries[0]}.png"))


def main(_):
    """
    Function to run TF Estimator
    Note: set the `TF_CONFIG` environment variable according to:
    https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate
    """

    ###################################
    # Set parameters/config
    ###################################

    model_params = {
        "n_classes": 2,
        "class_names": "not_object,object",
        "img_dim": 400,
        "num_channels": 3,
        "train_steps": 6000,
        "dense_size_a": 153,
        "dense_size": 153,
        "dense_activation": "relu",
        "dense_dropout_rate_a": 0.34,
        "dense_dropout_rate": 0.34,
        "optimizer": "adam",
        "metrics": [
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            FBetaScore(FLAGS.n_classes, beta=2.0, average="weighted"),
        ],
        "learning_rate": FLAGS.tf_learning_rate,
        "loss": tf.keras.losses.BinaryCrossentropy(),
    }

    def precision_m(labels, predictions):
        precision_metric = tf.keras.metrics.Precision(name="precision_m")
        precision_metric.update_state(
            y_true=labels, y_pred=predictions["output"]
        )
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

    ###################################
    # Check if user wants to run test
    ###################################
    # Create test dataset function if needed
    if FLAGS.tf_test_ckpt_path:
        classifier = resnet50_estimator(
            model_params, FLAGS.tf_test_ckpt_path, None
        )
        classifier = tf.estimator.add_metrics(classifier, fbeta_m)
        classifier = tf.estimator.add_metrics(classifier, precision_m)
        classifier = tf.estimator.add_metrics(classifier, recall_m)
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
        map_func = partial(
            parse_fn, n_chan=3, n_classes=model_params["n_classes"]
        )

        dataset_test = get_dataset_feeder(
            file_patterns=test_file_patterns,
            data_map_func=map_func,
            shuffle_buffer_size=0,
            repeat=False,
            n_map_threads=FLAGS.n_map_threads,
            batch_size=1,  # Use bs=1 here to count samples instead of batches
            prefetch_buffer_size=FLAGS.prefetch_buffer_size,
        )

        y_true = [features[1].numpy()[0] for features in dataset_test]
        print(f"Found {len(y_true)} total samples to test.")

        # Reset the dataset iteration for prediction
        dataset_test_fn = partial(
            get_dataset_feeder,
            file_patterns=test_file_patterns,
            data_map_func=map_func,
            shuffle_buffer_size=0,
            repeat=False,
            n_map_threads=FLAGS.n_map_threads,
            batch_size=10,
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
        ####################
        # plotting roc curve
        ###################
        y_test_arr = np.array(y_true)[:, -1]  # only select object column
        y_pred_arr = np.array(preds)[:, -1]
        print(y_test_arr.shape, y_pred_arr.shape)
        countries = FLAGS.countries

        plot_roc(y_test_arr, y_pred_arr, FLAGS.tf_test_results_dir, countries)

        ####################
        # writing result as CSVs
        ###################

        df_pred = pd.DataFrame.from_dict(output_d)
        df_pred.to_csv(os.path.join(FLAGS.tf_test_results_dir, "preds.csv"))

        print("preds csv written")

        recall_scores = [
            recall_score(np.array(y_true)[:, i], np.array(preds)[:, i])
            for i in np.arange(0, int(model_params["n_classes"]))
        ]

        precision_scores = [
            precision_score(np.array(y_true)[:, i], np.array(preds)[:, i])
            for i in np.arange(0, int(model_params["n_classes"]))
        ]

        fbeta_scores = [
            fbeta_score(np.array(y_true)[:, i], np.array(preds)[:, i], beta=2)
            for i in np.arange(0, int(model_params["n_classes"]))
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


if __name__ == "__main__":
    app.run(main)
