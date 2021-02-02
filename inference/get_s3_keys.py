"""
Getting tile list from S3 bucket
Author: @developmentseed
Use:
    python get_s3_keys.py --profile_name=aisurvey-nana \
                          --s3_bucket=aisurvey \
                          --s3_dir_prefix=cormon2019_chips \
                          --out_txt=tiles.txt
"""

import os
from os import path as op
import boto3
import click


@click.command(short_help="get tiles from s3 buckets")
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
    "--s3_bucket",
    help="s3 bucket name",
    required=True,
    type=str,
    default="aisurvey",
)

@click.option(
    "-d",
    "--s3_dir_prefix",
    help="s3 bucket name",
    required=True,
    type=str,
    default="cormon2019_chips",
)

@click.option(
    "-t",
    "--out_txt",
    help="text file name",
    required=True,
    type=str,
    default="tiles.txt",
)
def main(profile_name, s3_bucket, s3_dir_prefix, out_txt):
    session=boto3.Session(profile_name=profile_name)
    client=session.client('s3')
    paginator = client.get_paginator('list_objects')
    results =paginator.paginate(Bucket=s3_bucket,Prefix=s3_dir_prefix)
    with open(out_txt, 'w') as s3_txt:
        for i, result in enumerate(results):
                for j, s3_file in enumerate(result.get('Contents', [])):
                        print(s3_file)
                        fname = s3_file.get('Key')
                        print(f'{(i+1)*(j+1)}\n')
                        print(f'\nDownloaded {fname}')
                        s3_txt.write(f'{fname}\n')
                        print("*"*60)


if __name__ == '__main__':
    main()
