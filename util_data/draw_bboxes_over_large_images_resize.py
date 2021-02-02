"""
Script to draw bbox over images for labels' sanity check.

Author: @developmentseed

Run:
    python3 draw_bboxes_over_large_images_resize.py --csv=labeled_aiaia.csv

"""

from os import path as op
import json
from PIL import Image, ImageDraw
from PIL import ImageColor
import pandas as pd
import click

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
    return ((int(box[1]), int(box[0])), (int(box[3]), int(box[2])))


def class_color(c):
    # Taken from https://github.com/CartoDB/CartoColor/blob/master/cartocolor.js#L1633-L1733
    """Return 3-element tuple containing rgb values for a given class"""
    colors = ["#DDCC77", "#CC6677", "#117733", "#332288", "#AA4499", "#88CCEE"]
    if c == 0:
        return (0, 0, 0) # background class
    return ImageColor.getrgb(colors[c % len(colors)])

@click.command(short_help="Draw bounding boxes over the labels")
@click.option('--csv', help='csv that saved label, bbox, and normalized bboxes')

def main(csv, resize = (1024, 1024), label_text=False):
    """Draw bbox and label text over labels and colored by the categories
    Args:
        csv: csv that contains label, category, bboxes and normalized bboxes
        resize(tuple): new shape/size of the images
    Returns (None): RGB aerial image saved as png that have label bboxes drew
    """
    df = pd.read_csv(csv)
    df = characters_to_numb(df, 'category')
    images = pd.Series(df['image_id']).unique()
    for image in images:
        df_img = df[df['image_id']==image]
        df_img['bbox_norm'] = df_img.bbox_norm.apply(json.loads)
        img_name = op.basename(image)
        new_size = resize
        w, h = get_img_shape(image)
        with Image.open(image) as im:
            im_resize = im.resize(new_size)
            draw = ImageDraw.Draw(im_resize)
        attributes = [(bbox, label, labelID) for bbox, label, labelID
                       in zip(df_img.bbox_norm,df_img.label, df_img.category_id)]

        for attri in attributes:
            bbox = bbox_int([int(box*1024) for box in attri[0]])
            print((bbox[0], bbox[1]))
            draw.rectangle(bbox, outline=class_color(int(attri[2])), width=1)
            #comment the following script out if label text is not needed
            if label_text:
                draw.text(xy=bbox[1], text=str(attri[1]), fill='red', align="right")
            im_resize.save(f'{img_name}_large_bboxes.png', "PNG")

if __name__=="__main__":
    main()
