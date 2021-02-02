"""
Paralleling tile list from S3 bucket from classification for object detection model inference
Author: @developmentseed
Use:
    python sync_s3_keys_from_classifier.py \
                          --df_name=results_classification_0.5.csv \
                          --threshold=0.99 \
                          --profile_name=aisurvey-nana \
                          --bucket=aisurvey \
                          --dest_dir=chips
"""

import os
from os import path as op
import pandas as pd
import boto3
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import click


def download_s3keys_by_confident_score(tile, s3_profile_name, bucket, dest_dir):
    session = boto3.Session(profile_name=s3_profile_name)
    client = session.client('s3')
    dest_pathname = op.join(dest_dir, tile)
    if not op.exists(op.dirname(dest_pathname)):
        os.makedirs(op.dirname(dest_pathname))
    if op.exists(op.dirname(dest_pathname)):
        next
    client.download_file(bucket, tile, dest_pathname)

@click.command(short_help="get tiles from s3 buckets")
@click.option(
    "-d",
    "--df_name",
    help="Path and name of the dataframe from classificion inference",
    required=True,
    type=str,
    default="classifer.csv",
)

@click.option(
    "-t",
    "--threshold",
    help="Model confident score to filter the result from classification inference",
    required=True,
    type=str,
    default="aisurvey",
)

@click.option(
    "-p",
    "--profile_name",
    help="s3 profile",
    required=True,
    type=str,
    default="aisurvey-nana",
)

@click.option(
    "-b",
    "--bucket",
    help="s3 bucket name",
    required=True,
    type=str,
    default="aisurvey",
)


@click.option(
    "-p",
    "--dest_dir",
    help="the destination directory of images",
    required=True,
    type=str,
    default="chips",
)

def main(df_name, threshold, profile_name, bucket, dest_dir):
    df_raw= pd.read_csv(df_name)
    df = df_raw[df_raw['yes']>=float(threshold)]
    print(df.head())
    Parallel(n_jobs=-1)(delayed(download_s3keys_by_confident_score)(tile, profile_name, bucket, dest_dir)
                           for tile in tqdm(df['tile'], desc=f'Downloaded tile from S3 bucket {bucket}'))

if __name__ == '__main__':
    main()
