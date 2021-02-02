"""count the number labels and categories labeled

 Author: @developmentseed

 Run:
    python3 labels_counter.py --csv=labeled_aiaia.csv
"""

import os
from os import makedirs, path as op
import pandas as pd
import json
import click

def read_class_map():
    df = pd.read_csv('./../config/class_map.csv')
    class_id = {label[0]: 0 for label in zip(df.label)}
    return class_id

@click.command(short_help="count the number labels and categories labeled")
@click.option('--csv', help='csv that saved label, bbox, and normalized bboxes')

def counter(csv):
    df = pd.read_csv(csv)
    df['label'] = df.label.apply(lambda x: x.lower())
    map_dict = read_class_map()
    # df_aggr.to_csv('test.csv')
    for index, row in df.iterrows():
        map_dict[row['label']] += 1

    for d in map_dict:
      print(f'{d},{map_dict[d]}')

if __name__=="__main__":
    counter()
