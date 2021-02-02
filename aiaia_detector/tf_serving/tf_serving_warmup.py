"""
Script to generate Warmup requests.

Author: @developmentseed

Run:
    python3 tf_serving_warmup.py \
        --output=data/human_activities/001/assets.extra/tf_serving_warmup_requests

"""

import tensorflow as tf
import requests
import base64
import requests
import click
from tensorflow.python.framework import tensor_util
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2

IMAGE_URL = "https://ds-data-projects.s3.amazonaws.com/ai4earth/RR19_5EL_20180920A_RSO-L_1470_9_2.jpg"
NUM_RECORDS = 1000


def get_image_bytes():
    image_content = requests.get(IMAGE_URL, stream=True)
    image_content.raise_for_status()
    return image_content.content


@click.command(short_help="Script to generate Warmup requests")
@click.option(
    "-m",
    "--module_name",
    help="Module name",
    required=True,
    type=str,
    default="human_activities",
)
@click.option(
    "-o",
    "--output",
    help="Path for tf_serving_warmup_requests",
    required=True,
    type=str,
    default="data/human_activities/001/assets.extra/tf_serving_warmup_requests",
)
def main(module_name, output):
    image_bytes = get_image_bytes()
    with tf.io.TFRecordWriter(output) as writer:
        predict_request = predict_pb2.PredictRequest()
        predict_request.model_spec.name = module_name
        predict_request.model_spec.signature_name = "serving_default"
        predict_request.inputs["inputs"].CopyFrom(
            tensor_util.make_tensor_proto([image_bytes], tf.string)
        )
        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=predict_request)
        )
        for r in range(NUM_RECORDS):
            writer.write(log.SerializeToString())


if __name__ == "__main__":
    main()
