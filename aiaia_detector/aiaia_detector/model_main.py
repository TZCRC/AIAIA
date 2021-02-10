# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Binary to run train and evaluation on object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app, flags, logging

import tensorflow as tf

from object_detection import model_hparams
from object_detection import model_lib
import exporter
from object_detection.protos import pipeline_pb2

from google.protobuf import text_format


flags.DEFINE_string(
    "model_dir",
    None,
    "Path to output model directory "
    "where event and checkpoint files will be written.",
)
flags.DEFINE_string(
    "pipeline_config_path", None, "Path to pipeline config " "file."
)
flags.DEFINE_integer("num_train_steps", None, "Number of train steps.")
flags.DEFINE_boolean(
    "eval_training_data",
    False,
    "If training data should be evaluated for this job. Note "
    "that one call only use this in eval-only mode, and "
    "`checkpoint_dir` must be supplied.",
)
flags.DEFINE_integer(
    "sample_1_of_n_eval_examples",
    1,
    "Will sample one of " "every n eval input examples, where n is provided.",
)
flags.DEFINE_integer(
    "sample_1_of_n_eval_on_train_examples",
    5,
    "Will sample "
    "one of every n train input examples for evaluation, "
    "where n is provided. This is only used if "
    "`eval_training_data` is True.",
)
flags.DEFINE_string(
    "hparams_overrides",
    None,
    "Hyperparameter overrides, "
    "represented as a string containing comma-separated "
    "hparam_name=value pairs.",
)
flags.DEFINE_string(
    "checkpoint_dir",
    None,
    "Path to directory holding a checkpoint.  If "
    "`checkpoint_dir` is provided, this binary operates in eval-only mode, "
    "writing resulting metrics to `model_dir`.",
)
flags.DEFINE_boolean(
    "run_once",
    False,
    "If running in eval-only mode, whether to run just "
    "one round of eval vs running continuously (default).",
)

################################################################################
# Arguments for export the modules
################################################################################
flags.DEFINE_string(
    "input_type",
    "image_tensor",
    "Type of input node. Can be "
    "one of [`image_tensor`, `encoded_image_string_tensor`, "
    "`tf_example`] but use encoded_image_string_tensor for tfserving image containeriazation",
)

flags.DEFINE_string(
    "input_shape",
    None,
    "If input_type is `image_tensor`, this can explicitly set "
    "the shape of this input tensor to a fixed size. The "
    "dimensions are to be provided as a comma-separated list "
    "of integers. A value of -1 can be used for unknown "
    "dimensions. If not specified, for an `image_tensor, the "
    "default shape will be partially specified as "
    "`[None, None, None, 3]`.",
)
# flags.DEFINE_string('pipeline_config_path', None,
#                     'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
#                     'file.')
# flags.DEFINE_string('trained_checkpoint_prefix', None,
#                     'Path to trained checkpoint, typically of the form '
#                     'path/to/model.ckpt')
flags.DEFINE_string(
    "output_directory", None, "Path to write outputs - the export module."
)
flags.DEFINE_string(
    "config_override",
    "",
    "pipeline_pb2.TrainEvalPipelineConfig "
    "text proto to override pipeline_config_path.",
)
flags.DEFINE_boolean(
    "write_inference_graph", False, "If true, writes inference graph to disk."
)

FLAGS = flags.FLAGS


def trainer():
    flags.mark_flag_as_required("model_dir")
    flags.mark_flag_as_required("pipeline_config_path")
    config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir, keep_checkpoint_max=500
    )  # XXX: Manually adding `keep_checkpoint_max`

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
        pipeline_config_path=FLAGS.pipeline_config_path,
        train_steps=FLAGS.num_train_steps,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=(
            FLAGS.sample_1_of_n_eval_on_train_examples
        ),
    )
    estimator = train_and_eval_dict["estimator"]
    train_input_fn = train_and_eval_dict["train_input_fn"]
    eval_input_fns = train_and_eval_dict["eval_input_fns"]
    eval_on_train_input_fn = train_and_eval_dict["eval_on_train_input_fn"]
    predict_input_fn = train_and_eval_dict["predict_input_fn"]
    train_steps = train_and_eval_dict["train_steps"]

    if FLAGS.checkpoint_dir:
        if FLAGS.eval_training_data:
            name = "training_data"
            input_fn = eval_on_train_input_fn
        else:
            name = "validation_data"
            # The first eval input will be evaluated.
            input_fn = eval_input_fns[0]
        if FLAGS.run_once:
            estimator.evaluate(
                input_fn,
                steps=None,
                checkpoint_path=tf.train.latest_checkpoint(
                    FLAGS.checkpoint_dir
                ),
            )
        else:
            model_lib.continuous_eval(
                estimator, FLAGS.checkpoint_dir, input_fn, train_steps, name
            )
    else:
        train_spec, eval_specs = model_lib.create_train_and_eval_specs(
            train_input_fn,
            eval_input_fns,
            eval_on_train_input_fn,
            predict_input_fn,
            train_steps,
            eval_on_train_data=False,
        )

        # Currently only a single Eval Spec is allowed.
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])


def exportor():
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(FLAGS.pipeline_config_path, "r") as f:
        text_format.Merge(f.read(), pipeline_config)
    text_format.Merge(FLAGS.config_override, pipeline_config)
    if FLAGS.input_shape:
        input_shape = [
            int(dim) if dim != "-1" else None
            for dim in FLAGS.input_shape.split(",")
        ]
    else:
        input_shape = None
    # Loook for the latest checkpoint in model_dir
    trained_checkpoint_prefix = tf.train.latest_checkpoint(FLAGS.model_dir)
    exporter.export_inference_graph(
        FLAGS.input_type,
        pipeline_config,
        trained_checkpoint_prefix,
        FLAGS.output_directory,
        input_shape=input_shape,
        write_inference_graph=FLAGS.write_inference_graph,
    )


def main(unused_argv):
    ##########
    # running model training
    ##########
    import os

    print("tmp dir")
    print(os.listdir("/tmp"))
    print("Model Main dir")
    print(os.listdir("./"))
    print("top level dir")
    print(os.listdir("../"))
    if not os.path.exists("../logs"):
        os.makedirs("../logs")
    logging.get_absl_handler().use_absl_log_file("absl_logging", "../logs")
    model_dir = FLAGS.model_dir
    logging.info("Aiaia detector starts training")
    # trainer()
    logging.info("Aiaia detector finished training")
    ##########
    # running model exportation
    ##########
    logging.info("Aiaia detector model exporting")
    logging.info(f"from {model_dir}")
    exportor()
    logging.info("Aiaia detector finished model exportation")


if __name__ == "__main__":
    tf.app.run()
