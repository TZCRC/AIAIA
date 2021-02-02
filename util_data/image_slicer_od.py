"""
Util script to slide images and bboxes for object detection
The script will create new image chips called "tiles"
and a new csv that save all image chips information
called "RR19_train_sliced_image_nbboxes.csv"

Author: @developmentseed
Run:
    python3 image_slicer_od.py --in_csv=RR19_train.csv \
            --dimention=400 \
            --out_csv=RR19_train_sliced_image_nbboxes.csv \
            --out_tile_dir=RR19_tiles
"""

import os
from os import makedirs, path as op
from PIL import Image
import numpy as np
import pandas as pd
from math import floor
import json
import click


def bbox_within(bbox1, bbox2):
    """ finding relation between bbox1 and bbox2
    note: bbox1 is larger than bbox2

    Args:
        bbox1 (list): a list contain four corners of a chip/bbox;
        bbox2 (list): a list contain four corners of bbox;
    Return:
        bbox (list): a new bbox
    """

    x0, y0, x1, y1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    n0, m0, n1, m1 = round(bbox2[0]), round(bbox2[1]), round(bbox2[2]), round(bbox2[3])

    if np.logical_and(x0<n0<x1, y0<m0<y1) and np.logical_and(x0<n1<x1, y0<m1<y1):
        return bbox2

    # the second half of bbox inside of the chip
    if np.logical_and(x0<n1<x1, y0<m1<y1):
        # the top side-1
        if np.logical_and(x0<n0<x1, m0<y0<y1):
            return [n0, y0, n1, m1]
         #at the top left cornner- 5
        elif np.logical_and(n0<x0<x1, m0<y0<y1):
            return [x0, y0, n1, m1]
        # at the left side- 2
        else:
            return [x0, m0, n1, m1]

    # the first half of bbox inside of the chip
    if np.logical_and(x0<n0<x1, y0<m0<y0):
        # at right side - 3
        if np.logical_and(x0<n1<x1, y0<y1<m1):
            return [n0, m0, n1, y1]
        # at bottom right - 6
        elif np.logical_and(x0<<x1<n1, y0<y1<m1):
            return [n0, m0, x1, y1]
        # at the bottom side - 4
        else:
            return [n0, m0, x1, m1]

    # the first half of bbox inside of the top right corner -7
    if np.logical_and(n0<x0<x1, m0<y0<y1):
        if np.logical_and(x0<x1<n1, y0<m1<y1):
            return [n0, y0, x1, m1]

    # the second half of the bbox inside of the bottom left corner-8
    if np.logical_and(n0<x0<x1, y0<m0<y1):
        if np.logical_and(x0<n1<x1, y0<y1<m1):
            return [x0, m0, n1, y1]

def slice_img(img, dimention, out_tile_dir, save=True):
    """Slice image(if exist) into a given dimention chips

    Args:
        img (str): image path;
        dimention (int): dimention of the new chip, e.g. 400
        out_tile_dir: the directory to save new image chips
    Return:
        tiles (list): a list cotain tile name and tile bbox

    """
    tiles = []
    if os.path.exists(img):
        image = Image.open(img)
        img_name = op.basename(img)
        img_name = img_name.split('.')[0]
        width, height = image.size
        rows = floor(width/dimention)
        cols =  floor(height/dimention)
        print(f'it total {rows*cols} tiles with {rows} rows and {cols} columns')

        for i in range(cols):
            for j in range(rows):
                left, upper, right, lower = (j*dimention, i*dimention, (j+1)*dimention, (i+1)*dimention)
                image_crop = image.crop((left, upper, right, lower))
                tile_name = f'{img_name}_{j}_{i}.jpg'

                if save:
                    if not op.isdir(out_tile_dir):
                        makedirs(out_tile_dir)
                    tile_path = op.join(out_tile_dir,tile_name)
                    print(f'{tile_path} saved!')
                    image_crop.save(tile_path)
                tiles.append([tile_name, [left, upper, right, lower]])
    return tiles

def get_image_attributes(df, img):
    """extract attributes(id, label, bbox, category) of a given image from df

    ------df.head(1)-------
    image_id, label, bbox, bbox_norm, category
    TA25-RKE-20191202A/TA25-RKE_001.jpg, oryx, [2653.03, 5085.03, 2731.24, 5122.82], [0.663, 0.845, 0.683, 0.852], wildlife
    ----------------
    Args:
        df: pandas dataframe
        img: a given image name/path

    Rturn:
         attributes (list): a list contain bbox, label and category for a given image
    """
    df_image = df[df['image_id']==img]
    df_image['bbox'] = df_image.bbox.apply(json.loads)
    attributes = [(bbox, label, categ) for bbox, label, categ in
                  zip(df_image.bbox, df_image.label, df_image.category)]
    return attributes

def shift_bbox(im_name,tile_bbox, bbox):
    """ after large image slice by row and col, shift bbox all
    together to be based off 400 x 400

    Args:
        im_name: image chip name. To extract row and col
        tile_bbox: four corners of image chips;
        bbox: four corners of bbox for objects
    Return:
        new_bbox: a shift objects base off 400x400 image chip
    """
    name, ext = op.splitext(im_name)
    row = name.split('_')[-2]
    col = name.split('_')[-1]
    xmin, ymin, xmax, ymax = tile_bbox[0], tile_bbox[1], tile_bbox[2], tile_bbox[3]
    nmin, mmin, nmax, mmax = bbox[0], bbox[1], bbox[2], bbox[3]
    n0, m0, n1, m1= round(nmin - xmin), round(mmin-ymin), round(nmax-xmin), round(mmax-ymin)
    arr = np.where(np.array([n0, m0, n1, m1])<=0, 0, np.array([n0, m0, n1, m1]))
    new_bbox = list(arr)
    return new_bbox


@click.command(short_help="slice image and bboxe by image chips for training data creation ")
@click.option('--in_csv', help='csv that saved label, bbox, and normalized bboxes')
@click.option('--dimention', help='dimention of the image, e.g. 400')
@click.option('--out_csv', help='csv that will save image chips label, bbox, and normalized bboxes')
@click.option('--out_tile_dir', help='output path to save all the tiles/image chips')

def image_bbox_slicer(in_csv, dimention, out_csv, out_tile_dir, save=True):
    """slice image into chips by given dimention

    Args:
        csv(str): path of csv;
        dimention(int): chip dimention;
        save (bool): if the chip is save to a directory called 'tiles'
    Return:
        (None): chips save in 'tiles' directory and a csv 'tiles_aiaia_sliced_image_nbboxes.csv'
    ------------tiles_aiaia_sliced_image_nbboxes.csv----------
    tile_id,org_bbox,bbox,label,category
    TA25-RKE-20191202A_L_4886_6_4.jpg,[2688.57, 1862.77, 2768.26, 1935.82],[289, 263, 368, 336],oryx,wildlife
    ----------------------------------------------------------
    """
    df = pd.read_csv(in_csv)
    tile_bboxes = []
    images = pd.Series(df['image_id']).unique()
    tile_bboxes = []
    dimention = int(dimention)

    for img in images:
        tiles = slice_img(img, dimention, out_tile_dir, save)
        if len(tiles) > 0:
            attributes = get_image_attributes(df, img)
            for tile in tiles:
                for attri in list(attributes):
                    bbox_return = bbox_within(tile[1], attri[0])
                    if bbox_return is not None:
                        bbox_shift = shift_bbox(tile[0],tile[1], bbox_return)
                        print(f'tile {tile[1]} contains {attri[0]} and new box is {bbox_shift}!')
                        tile_bboxes.append([tile[0], attri[0], bbox_shift, attri[1], attri[2]])

    col_names = ['tile_id', 'org_bbox', 'bbox', 'label', 'category']
    df_tiles = pd.DataFrame(tile_bboxes, columns=col_names)
    print(f"total {len(df_tiles)} tiles with bboxes saved as csv {out_csv}")
    df_tiles.to_csv(out_csv)

if __name__=="__main__":
    image_bbox_slicer()
