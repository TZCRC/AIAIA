"""
Script to create CVAT xml file from inference files and geloccation files, reprojecting to original images ai4 earth

Author: @developmentseed

Run:
    python3 prediction2cvatxml.py \
    --csv_location_file=data/data/ai4earth_locations.csv \
    --json_prediction_file=data/inference_results.json \
    --category=human_activities \
    --threshold=0.85 \
    --output_xml_file=data/human_activities_inference_results.xml
"""

from os import path as op
import json
import click
from smart_open import open
import pandas as pd
import xml.etree.ElementTree as ET
from xml.etree import ElementTree
from xml.dom import minidom

from post_util import df_to_geojson, filter_predictions, prettify


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


def json2xml(list_predictions):
    root = ET.Element("annotations")
    version = ET.SubElement(root, "version")
    version.text = str("1.1")
    # Sort list of predition by image name
    list_predictions = sorted(
        list_predictions, key=lambda p: op.join(p["sourcefile"], p["fname"])
    )

    for index, preds in enumerate(list_predictions):
        image_path = op.join(preds["sourcefile"], preds["fname"])
        image = ET.SubElement(root, "image")
        image.attrib["id"] = str(index)
        image.attrib["name"] = image_path
        image.attrib["width"] = str(preds["imagesize"][0])
        image.attrib["height"] = str(preds["imagesize"][1])
        for pred in preds["predictions"]:
            for idx, class_name in enumerate(pred["classes"]):
                # Add bbox
                box = ET.SubElement(image, "box")
                box.attrib["label"] = str("wildlife")
                box.attrib["occluded"] = str("0")
                ytl, xtl, ybr, xbr = pred["boxes"][idx]
                box.attrib["xtl"] = str(xtl)
                box.attrib["ytl"] = str(ytl)
                box.attrib["xbr"] = str(xbr)
                box.attrib["ybr"] = str(ybr)
                attribute = ET.SubElement(box, "attribute")
                attribute.attrib["name"] = str("wildlife")
                attribute.text = str(class_name)
    return root


def fix_bbox(preds):
    # These values  need to be change for other projects with different size of image
    chip_width = 400
    chip_height = 400
    for pred in preds["predictions"]:
        chip = op.splitext(op.basename(pred["image_id"]))[0]
        chip_position = list(map(int, chip.split("_")[-2:]))
        # Chip position on the image
        chip_position_x = chip_position[0] + 1
        chip_position_y = chip_position[1] + 1
        for i, bbox in enumerate(pred["boxes"]):
            xtl, ytl, xbr, ybr = bbox
            # Multiply the box result by 400 and sum up the value of three coordinates on the image
            new_xtl = xtl * chip_width + chip_width * chip_position_x
            new_ytl = ytl * chip_height + chip_height * chip_position_y
            new_xbr = xbr * chip_width + chip_width * chip_position_x
            new_ybr = ybr * chip_height + chip_height * chip_position_y
            pred["boxes"][i] = [new_xtl, new_ytl, new_xbr, new_ybr]
    return preds


@click.command(
    short_help="Script to match predictions with the geo-coordinates"
)
@click.option(
    "--csv_location_file",
    help="Location for images",
    required=True,
    type=str,
    default="data/mxj2019.csv",
)
@click.option(
    "--json_prediction_file",
    help="Prediction json file",
    required=True,
    type=str,
    default="data/wildlife_inference_results.json",
)
@click.option(
    "--category",
    help="Category",
    required=True,
    type=str,
    default="wildlife",
)
@click.option(
    "--output_xml_file",
    help="Output xml file",
    required=True,
    type=str,
    default="data/wildlife_results.xml",
)
@click.option(
    "--threshold",
    help="threshold for filter data",
    required=True,
    type=float,
    default=0.85,
)
def main(
    csv_location_file,
    json_prediction_file,
    category,
    threshold,
    output_xml_file,
):

    ###################################################################
    # Select category
    ###################################################################

    classes_dict = {}
    if category == "wildlife":
        classes_dict = wildlife_classes
    elif category == "livestock":
        classes_dict = livestock_classes
    elif category == "human_activities":
        classes_dict = human_activities_classes

    ###################################################################
    # Read CSV
    ###################################################################

    df = pd.read_csv(csv_location_file)

    # ###################################################################
    # # Read predictions json file
    # ###################################################################
    with open(json_prediction_file) as f:
        predictions = json.load(f)

    # Filter predictions
    predictions = filter_predictions(predictions, threshold)
    dict_predictions = {}
    for pred in predictions:
        fname = op.basename(pred["image_id"])
        original_fname = fname.rsplit("_", 2)[0] + ".jpg"
        if len(pred["classes"]) > 0:

            if dict_predictions.get(original_fname, "*") == "*":
                dict_predictions[original_fname] = {"predictions": [pred]}
            else:
                dict_predictions[original_fname]["predictions"].append(pred)

    matched_predictions = []
    for key, preds in dict_predictions.items():
        df_ = df.loc[df["fname"] == key]
        if df_.shape[0] >= 1:
            df_row = df_.iloc[0]
            # Recolect all the data from csv
            preds["fname"] = df_row["fname"]
            preds["imagesize"] = list(map(int, df_row["imagesize"].split("x")))
            preds["sourcefile"] = op.dirname(df_row["sourcefile"])
            preds["x"] = df_row["x"]
            preds["y"] = df_row["y"]
            preds = fix_bbox(preds)
            matched_predictions.append(preds)

    # Save on xml
    xml_root = json2xml(matched_predictions)
    f = open(output_xml_file, "w")
    f.write(prettify(xml_root))
    f.close()


if __name__ == "__main__":
    main()
