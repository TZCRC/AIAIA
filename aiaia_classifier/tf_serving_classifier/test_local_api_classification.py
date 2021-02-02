"""
Reading images from S3 and test model tfserving cpu RESTful API locally

Author: @developmentseed

Use:
    python test_local_api_classification.py --model_name=xception_classifier \
                                            --s3_profile=default --batch_size=10
"""

import os
from os import path as op
from itertools import zip_longest
from typing import List, Dict, Any
import boto3
import json
import base64
import requests
from os import makedirs, path as op
import time
import click
from tqdm import tqdm
start_time = time.time()

def _grouper(iterable, n, fillvalue=None):
    "Itertool recipe to collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def get_images(s3_profile, s3_keys: List, download_folder):
    if not op.isdir(download_folder):
        makedirs(download_folder)
    files = []
    s3 = boto3.Session(profile_name=s3_profile).client("s3")
    for s3_file in s3_keys:
        filename = download_folder + "/" + s3_file.split("/")[1]
        s3.download_file("aisurvey", s3_file, filename)
        files.append(filename)
    return files

def post_prediction(url_endpoint, payload):
    resp = requests.post(url_endpoint, data=payload)
    resp.raise_for_status()
    return resp.json()

@click.command(short_help="Testing local tfserving api")
@click.option(
    "-m",
    "--model_name",
    help="Model name",
    required=True,
    type=str,
    default="livestock",
)
@click.option(
    "-s",
    "--s3_profile",
    help="S3 profile",
    required=True,
    type=str,
    default="ai4e",
)
@click.option(
    "-b",
    "--batch_size",
    help="batch_size",
    required=True,
    type=str,
    default=5,
)

def main(model_name, s3_profile, batch_size = 5):
    # Set the url of the running Docker container
    # url_endpoint = "http://ai4ea-ai4ea-12guawqsr5g2p-176578657.us-east-1.elb.amazonaws.com/v1/models/wildlife:predict"
    url_endpoint = f"http://localhost:8501/v1/models/{model_name}:predict"

    s3 = boto3.Session(profile_name=s3_profile).client("s3")
    s3_keys = [
        "cormon2019_chips/wcm_n51_L_20190602095318_14_8.jpg",
        "cormon2019_chips/wcm_n51_L_20190602095318_14_9.jpg",
        "cormon2019_chips/wcm_n51_L_20190602095318_9_9.jpg",
        "cormon2019_chips/wcm_n51_L_20190602095318_9_8.jpg",
    ]

    # images = get_images(s3_profile, s3_keys, "data/images")
    image_directory = "test_data"
    images = [op.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(".jpg")]
    # # Iterate through groups of images
    for i, img_group in enumerate(_grouper(images, int(batch_size))):
        print(img_group)
        instances = []
        for batch_img_fname in img_group:
            if batch_img_fname is not None:
                with open(batch_img_fname, 'rb') as image_file:
                    b64_image = base64.b64encode(image_file.read())
                    instances.append({'b64': b64_image.decode('utf-8')})
        load_file_time = time.time() - start_time
        print(f'Load files : {str(load_file_time)}')
        # # Run prediction
        payload = json.dumps({"instances": instances})
        content = post_prediction(url_endpoint, payload)
        print(f'post_prediction : {str(time.time() -load_file_time- start_time)}')
        print(json.dumps(content))

if __name__ == '__main__':
    main()
