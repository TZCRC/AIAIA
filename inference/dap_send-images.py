"""
dap_send-images.py

Populate an SQS queue from the gaint list of objects/images from S3 bucket.

@author:developmentseed

use:
python3 dap_send-images.py --imgs_txt_files=aiaia_inference_test_short.txt \
                           --sqs_url=https://sqs.us-xxx
"""

import numpy as np
import boto3
from tqdm import tqdm
from joblib import Parallel, delayed
import click


def send_img_msg_batch(queue, msg_template, batch_inds):
    """Send a batch of message."""
    entries = []
    for id_num, imgID in enumerate(batch_inds):
        entries.append({'MessageBody':msg_template.format(imgID=imgID),
                        'Id':str(id_num)})

    try:
        response = queue.send_messages(Entries=entries)

        if response.get('Failed'):
            print(response.get('Failed'))

    except Exception as e:
        print('Error in pushing tiles: {}. Error:\n{}'.format(entries, e))


def send_img_msg(queue, msg_template, imgID):
    """Send a single message to SQS queue."""
    msg_body = msg_template.format(str(imgID))
    try:
        response = queue.send_message(MessageBody=msg_body)

        if response.get('Failed'):
            print(response.get('Failed'))

    except Exception as e:
        print('Error in pushing tile: {}. Error:\n{}'.format(msg_body, e))
@click.command(short_help="Pushing tiles as SQS queue to Chip n Scale")
@click.option(
    "-ts",
    "--imgs_txt_files",
    help="text files seperate by ,",
    required=True,
    type=str,
    default="text1.txt, text2.txt",
)
@click.option(
    "-s",
    "--sqs_url",
    help="SQS url that print out when Chip n Scale deploy",
    required=True,
    type=str,
    default="https://sqs.us-east-1.amazonaws.com/552819999234/aiaiaTileQueue",
)
def main(imgs_txt_files, sqs_url):
    msg_template='{{"image":"{imgID}"}}'
    sqs = boto3.resource('sqs', region_name='us-east-1')
    queue =sqs.Queue(sqs_url)
    # Get tiles from text file if pre-computed
    with open(imgs_txt_files, 'r') as img_file:
        img_inds = [ind.strip() for ind in img_file if len(ind)]
    batch_size = 10
    batch_inds = len(img_inds) // batch_size + 1
    img_ind_batches = [img_inds[b1 * batch_size:b2 * batch_size]
                        for b1, b2 in zip(np.arange(0, batch_inds - 1),
                                          np.arange(1, batch_inds))]
    Parallel(16, prefer='threads')(delayed(send_img_msg_batch)(
    queue, msg_template, img_ind_batch) for img_ind_batch
    in tqdm(img_ind_batches))

if __name__=="__main__":
    main()
