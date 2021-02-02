"""
Script to draw bbox over images for labels' sanity check.

Author: @developmentseed

Run:
    python3 draw_bboxes_over_images.py \
        --csv=RR17_train_sliced_image_nbboxes.csv \
        --chip_dir=RR17_tiles \
        --out_chip_dir=RR17_tiles_inspect

"""

from os import makedirs, path as op
import json
from PIL import Image, ImageDraw
from PIL import ImageColor
import pandas as pd
import click
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

def characters_to_numb(df, cc):
    """categorize column text to code

    Args:
        df: pandas dataframe
        cc(str): column name, e.g. 'label'.

    Returns:
        df: pandas dataframe with new labels
    """
    df.cc_ = pd.Categorical(df[cc])
    df[f'{cc}_id'] = df.cc_.codes
    return df

def get_img_shape(img):
    """get width, height of an image

    Args:
        img(str): image path
    Returns:
        width, height (int): width and height of the image
    """
    image = Image.open(img)
    width, height = image.size
    return width, height

def bbox_int(box):
    return ((int(box[0]), int(box[1])), (int(box[2]), int(box[3])))

def norm_bbox(bbox, img_height, img_width):
    """normalize bbox to be betwween 0, 1

    Args:
        bbox (list): original labeled bounding box;
        img_width(int): image width;
        img_height(int): image height;
    Returns:
        norm_bbox: normolized bbox
    """
    ymin, xmin, ymax, xmax = bbox[0], bbox[1], bbox[2], bbox[3]
    b0 = ymin / img_height
    b1 = xmin / img_width
    b2 = ymax / img_height
    b3 = xmax / img_width
    return [b0, b1, b2, b3]

def multiple_bbox(bbox, img_height, img_width):
        """normalize bbox to be betwween 0, 1

        Args:
            bbox (list): original labeled bounding box;
            img_width(int): image width;
            img_height(int): image height;
        Returns:
            norm_bbox: normolized bbox
        """
        xmin, ymin, xmax, ymax= bbox[0], bbox[1], bbox[2], bbox[3]
        b0 = ymin * img_height
        b1 = xmin * img_width
        b2 = ymax * img_height
        b3 = xmax * img_width
        return [b0, b1, b2, b3]

def class_color(c):
    # Taken from https://github.com/CartoDB/CartoColor/blob/master/cartocolor.js#L1633-L1733
    """Return 3-element tuple containing rgb values for a given class"""
    colors = ["#DDCC77", "#CC6677", "#117733", "#332288", "#AA4499", "#88CCEE"]
    if c == 0:
        return (0, 0, 0) # background class
    return ImageColor.getrgb(colors[c % len(colors)])


def draw_bbox(image, df, chip_dir, out_chip_dir, label_text):
    """Draw the bbox on the images

    Args:
        img(str): image path
        df(Data frame):
        chip_dir(str): path for input images
        out_chip_dir(str): path for the output images
        label_text(bool): option to draw the label or not
    Returns:
        Save an image with bboxes drawed.
    """
    df_img = df[df['tile_id'] == image]
    image = op.join(chip_dir, image)
    df_img = df_img.copy()
    df_img['bbox'] = df_img.loc[:, "bbox"].apply(json.loads)
    attributes = [(bbox, label, labelID) for bbox, label, labelID
                  in zip(df_img.bbox, df_img.label, df_img.category_id)]
    img_name = op.basename(image).split('.')[0]
    w, h = get_img_shape(image)
    with Image.open(image) as im:
        draw = ImageDraw.Draw(im)

    for attri in attributes:
        bbox = bbox_int(attri[0])
        xtl = bbox[0][0]
        ytl = bbox[0][1]
        xbr = bbox[1][0]
        ybr = bbox[1][1]

        draw.rectangle(bbox, outline=class_color(int(attri[2])), width=2)
           #comment the following script out if label text is not needed
        if label_text:
            x_label = xtl + (xbr - xtl) / 2
            # y_label = ytl + (ybr - ytl) / 2
            draw.text(
                xy=(x_label - 15, ybr), text=str(attri[1]), fill="red", align="right"
            )
            # draw the coordinates
            draw.text(xy=(bbox[0]), text=str(bbox[0]), fill="yellow", align="center")
            draw.text(xy=bbox[1], text=str(bbox[1]), fill="yellow", align="center")

            # draw.text(xy=bbox[1], text=str(attri[1]), fill='red', align="right")
            im.save(f'{out_chip_dir}/{img_name}_bboxes.png', "PNG")

@click.command(short_help="Draw bounding boxes over the labels")
@click.option('--csv', help='csv that saved label, bbox, and normalized bboxes')
@click.option('--chip_dir', help='csv that saved label, bbox, and normalized bboxes')
@click.option('--out_chip_dir', help='Path to save RGB images with the bbox drawed')

def main(csv, chip_dir, out_chip_dir, label_text=True):
    """Draw bbox and label text over labels and colored by the categories
    Args:
        csv: csv that contains label, category, bboxes and normalized bboxes
        resize(tuple): new shape/size of the images
    Returns (None): RGB aerial image saved as png that have label bboxes drew
    """
    df = pd.read_csv(csv)
    df = characters_to_numb(df, 'category')
    images = pd.Series(df['tile_id']).unique()
    out_chip_dir = out_chip_dir.strip("/")
    if not op.isdir(out_chip_dir):
        makedirs(out_chip_dir)

    Parallel(n_jobs=-1)(
        delayed(draw_bbox)(image, df, chip_dir, out_chip_dir, label_text)
        for image in tqdm(images, desc=f'Drawing bbox on the images... '))

if __name__=="__main__":
    main()
