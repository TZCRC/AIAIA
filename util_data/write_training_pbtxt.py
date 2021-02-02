"""to get pbtxt for training data classes for TF Object Detection.
The class id is given by the quantity of labels.
For instance, cow has 4910 label, and shoats has 1906, so cow is 1 and shoats is 2.

-----pbtxt------
  "item": {
    "id": 3,
    "name": "buffalo"
  }
  "item": {
    "id": 4,
    "name": "wildebeest"
  }
  "item": {
    "id": 5,
    "name": "elephant"
  }
 ----------------

 Author: @developmentseed

 Run:
    python3 write_training_pbtxt.py --csv=labeled_aiaia.csv --out_txt=aiaia_pbtxt.txt
"""

import os
from os import makedirs, path as op
import pandas as pd
import json
import click


@click.command(short_help="write pbtxt from labels csv")
@click.option('--csv', help='csv class map')
@click.option('--out_dir', help='output directory')
def write_pbtxt(csv, out_dir):
    out_dir = out_dir.strip("/")
    if not op.isdir(out_dir):
        makedirs(out_dir)

    df = pd.read_csv(csv)
    df = df[df['label'] != 'crane']
    df = df[df['label'] != 'ostrich']
    df = df[df['label'] != 'stork']
    df = df[df['label'] != 'lion']
    df = df.drop(['label', 'label_id'], axis=1)
    df.rename(columns={'group': 'label'}, inplace=True)

    categories = ['human_activities', 'livestock', 'wildlife', 'master']
    for category in categories:
        df_fixed = df.copy()
        if category == 'master':
            df_fixed.rename(columns={'master_group_id': 'class_id'}, inplace=True)
        else:
            df_fixed = df_fixed[df_fixed['category'] == category]
            df_fixed.rename(columns={'group_id': 'class_id'}, inplace=True)

        pbtxt_dict = {}
        for _index, row in df_fixed.iterrows():
            pbtxt_dict[row['label']] = {
                'item': {
                    'id': int(row['class_id']),
                    'name': row['label']
                }
            }

        out_txt = f'{out_dir}/{category}.pbtxt'
        with open(out_txt, 'w') as aia:
            for (_i, obj) in pbtxt_dict.items():
                item = json.dumps(obj['item'])
                id = obj['item']['id']
                name = obj['item']['name']
                item_txtt = f'item: {{\n\tid: {id},\n\tname: "{name}" \n}}'
                aia.write(f'{item_txtt}\n')
        print(f'{out_txt} has been written to the disk!')

if __name__ == "__main__":
    write_pbtxt()
