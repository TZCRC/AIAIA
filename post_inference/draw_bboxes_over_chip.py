"""
Script to draw bbox over chips for prediction inspection

Author: @developmentseed

Run:
    python3 draw_bboxes_over_chip.py \
        --json_file=data/wildlife_results.json \
        --aws_profile=devseed \
        --out_chip_dir=data/wildlife_inspect

"""

import io
from os import makedirs, path as op
import json
from PIL import Image, ImageDraw
import click
from joblib import Parallel, delayed
from tqdm import tqdm
from smart_open import open
import random

wildlife_classes = {
    1: "buffalo",
    2: "dark_coloured_large",
    3: "elephant",
    4: "giraffe",
    5: "hippopotamus",
    6: "light_coloured_large",
    7: "smaller_ungulates",
    8: "warthog",
    9: "zebra",
}

livestock_classes = {1: "cow", 2: "donkey", 3: "shoats"}

human_activities_classes = {
    1: "boma",
    2: "building",
    3: "charcoal_mound",
    4: "charcoal_sack",
    5: "human",
}


def multiple_bbox(bbox, img_height, img_width):
    """normalize bbox to be betwween 0, 1

    Args:
        bbox (list): original labeled bounding box;
        img_width(int): image width;
        img_height(int): image height;
    Returns:
        norm_bbox: normolized bbox
    """
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    b0 = ymin * img_height
    b1 = xmin * img_width
    b2 = ymax * img_height
    b3 = xmax * img_width
    return [b0, b1, b2, b3]


def class_color(c):
    """color picker

    Args:
        c (int): init whicch represents the color

    Returns:
       str: color string code
    """
    # Taken from https://github.com/CartoDB/CartoColor/blob/master/cartocolor.js#L1633-L1733
    colors = [
        "#FF0000",
        "#FFC000",
        "#FFFC00",
        "#FF0000",
        "#00FFFF",
        "#FF0000",
        "#44AA99",
        "#999933",
        "#888888",
        "#DF5B5A",
    ]
    return colors[c]


def draw_bbox(pred, aws_bucket, out_chip_dir, category):
    """Draw the bbox on the images

    Args:
        pred(obj): image path
        aws_bucket(str):
        out_chip_dir(str): path for the output images
    Returns:
        Save an image with bboxes drawed.
    """
    # classes_dict={}
    if category == "wildlife":
        classes_dict = wildlife_classes
    elif category == "livestock":
        classes_dict = livestock_classes
    elif category == "human_activities":
        classes_dict = human_activities_classes
    # Hardcode, to remove root directories from the image patth on inference
    pred["image_id"] = (
        pred["image_id"]
        .replace("chips_all", "")
        .replace("chips_75", "")
        .replace("chips_sl25_rr19_75", "")
    )
    s3_link_img = f"s3://{aws_bucket}" + pred["image_id"]
    print(s3_link_img)
    if len(pred["classes"]) > 0:
        with open(s3_link_img, "rb") as f:
            image_binary = f.read()
            image_stream = io.BytesIO(image_binary)
            image_file = Image.open(image_stream)
            draw = ImageDraw.Draw(image_file)
            img_width, img_height = image_file.size
            for i, class_id in enumerate(pred["classes"]):
                class_id = int(class_id)
                bbox = pred["boxes"][i]
                bbox = multiple_bbox(bbox, img_height, img_width)
                xtl = int(bbox[0])
                ytl = int(bbox[1])
                draw.rectangle(bbox, outline=class_color(class_id), width=2)
                class_name = classes_dict[class_id]
                class_score = round(pred["scores"][i], 2)
                draw.text(
                    xy=(xtl, ytl - 10),
                    text=f"{class_name}-{class_score}",
                    fill="white",
                    align="center",
                )
            oput_img_path = op.join(out_chip_dir, op.basename(pred["image_id"]))
            image_file.save(oput_img_path)


@click.command(short_help="Draw bounding boxes over the labels")
@click.option(
    "--json_file",
    help="Prediction json file.[{item},{item}]",
    required=True,
    type=str,
    default="data/wildlife_results/result_408.json",
)
@click.option(
    "--category",
    help="Category",
    required=True,
    type=str,
    default="wildlife",
)
@click.option(
    "--aws_bucket",
    help="AWS bucket where are stored the images",
    required=True,
    type=str,
    default="aisurvey",
)
@click.option(
    "--out_chip_dir",
    help="Path to save RGB images with the bbox drawed",
    required=True,
    type=str,
    default="data/wildlife_inspect",
)
@click.option(
    "--random_sample",
    help="Percentage to get samples",
    required=True,
    type=float,
    default=0.2,
)
def main(json_file, category, aws_bucket, out_chip_dir, random_sample):

    with open(json_file) as f:
        predictions = json.load(f)

    out_chip_dir = out_chip_dir.strip("/")
    if not op.isdir(out_chip_dir):
        makedirs(out_chip_dir)

    num = int(len(predictions) * random_sample)
    random_predictions = random.sample(predictions, num)
    random.shuffle(random_predictions)

    Parallel(n_jobs=-1)(
        delayed(draw_bbox)(pred, aws_bucket, out_chip_dir, category)
        for pred in tqdm(
            random_predictions, desc=f"Drawing bbox on the images... "
        )
    )


if __name__ == "__main__":
    main()
