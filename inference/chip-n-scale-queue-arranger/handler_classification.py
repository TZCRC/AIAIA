"""
Example AWS Lambda function for chip-n-scale
To read images directly from S3 bucket.
Replace this handler.py to "chip-n-scale-queue-arranger/lambda/download_and_predict/handler.py"
author: @developmentseed
"""

import os
from os import path as op
import pg8000
from typing import List, Dict, Any
import boto3
import json
import base64

from download_and_predict.base import DownloadAndPredict
from download_and_predict.custom_types import SQSEvent

class S3_DownloadAndPredict(DownloadAndPredict):
    """
    base object DownloadAndPredict implementing all necessary methods to
    make machine learning predictions
    """

    def __init__(self, bucket: str, db: str, prediction_endpoint: str):
        super(DownloadAndPredict, self).__init__()
        self.bucket = bucket
        self.db = db
        self.prediction_endpoint = prediction_endpoint


    def get_images(self, s3_keys: List):
        s3_client = boto3.client('s3', aws_access_key_id=os.getenv('AISURVEY_AWS_ACCESS_KEY_ID'),
                                       aws_secret_access_key=os.getenv('AISURVEY_AWS_SECRET_ACCESS_KEY')) # using WB's credential or ask WB for an IAM role
        for s3_file in s3_keys:
            key = json.loads(s3_file)['image']
            response = s3_client.get_object(Bucket =self.bucket, Key = key)
            yield(key, response["Body"].read())


def handler(event: SQSEvent, context: Dict[str, Any]) -> None:
    # read all our environment variables to throw errors early
    bucket =os.getenv('BUCKET')
    db = os.getenv('DATABASE_URL')
    prediction_endpoint=os.getenv('PREDICTION_ENDPOINT')

    assert(bucket)
    assert(db)
    assert(prediction_endpoint)

    # instantiate our DownloadAndPredict class
    dap = S3_DownloadAndPredict(
        bucket=bucket,
        db=db,
        prediction_endpoint=prediction_endpoint
    )

    # construct a payload for our prediction endpoint
    s3_keys =[record['body'] for record in event['Records']]
    print(s3_keys)
    # TODO
    # do we need to read data directly from s3.
    tile_indices, payload = dap.get_prediction_payload(s3_keys)
    print('got images')

    # send prediction request
    content = dap.post_prediction(payload)
    print('got predictions')
    dap.save_to_db(
        tile_indices,
        content['predictions'],
        result_wrapper=lambda x: pg8000.PGJsonb(x)
    )
