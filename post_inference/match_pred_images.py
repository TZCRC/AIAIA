"""
Script to match predictions with the geo-coordinattes

Author: @developmentseed

Run:
    python3 match_pred_images.py \
        --csv_location_file=data/data/ai4earth_locations.csv \
        --json_prediction_file=data/human_activities_inference_results.json \
        --category=human_activities \
        --threshold=0.85 \
        --output_csv_file=data/human_activities_inference_results.csv \
        --output_geojson_file=data/human_activities_inference_results.geojson
"""

from os import path as op
import json
import click
from smart_open import open
import pandas as pd
from post_util import df_to_geojson, filter_predictions

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


@click.command(
    short_help="Script to match predictions with the geo-coordinates"
)
@click.option(
    "--csv_location_file",
    help="Location for images",
    required=True,
    type=str,
    default="data/ai4earth_locations.csv",
)
@click.option(
    "--json_prediction_file",
    help="Prediction json file",
    required=True,
    type=str,
    default="data/wildlife_results.json",
)
@click.option(
    "--category",
    help="Category",
    required=True,
    type=str,
    # default="wildlife",
)
@click.option(
    "--output_csv_file",
    help="Output csv file",
    required=True,
    type=str,
    default="data/wildlife_results.csv",
)
@click.option(
    "--output_geojson_file",
    help="Output geojson file",
    required=True,
    type=str,
    default="data/wildlife_results.geojson",
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
    output_csv_file,
    output_geojson_file,
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
                dict_predictions[original_fname] = [pred]
            else:
                dict_predictions[original_fname].append(pred)

    ###################################################################
    # Read predictions json file
    ###################################################################

    list_preds = []
    missing_location = []

    for key in dict_predictions:
        df_ = df.loc[df["fname"] == key]
        if df_.shape[0] >= 1:
            df_row = df_.iloc[0]
            # start in 0 all classes
            classes_on_img = dict(
                [(value, 0) for key, value in classes_dict.items()]
            )

            classes_on_img["fname"] = df_row["fname"]
            classes_on_img["sourcefile"] = op.dirname(df_row["sourcefile"])
            classes_on_img["x"] = df_row["x"]
            classes_on_img["y"] = df_row["y"]
            # Count objects per class in each image
            classes_in_img = []
            for pred in dict_predictions[key]:
                classes_in_img = classes_in_img + pred["classes"]

            num_classes = {
                classes_dict[i]: classes_in_img.count(i) for i in classes_in_img
            }
            num_classes["total"] = sum(num_classes.values())
            classes_on_img.update(num_classes)
            list_preds.append(classes_on_img)
        else:
            # print("No location for: " + key)
            missing_location.append(key)

    df_result = pd.DataFrame.from_dict(list_preds)
    df_result = df_result.fillna(0)

    df_result.to_csv(output_csv_file, index=False)

    # Save csv as geojson
    columns = list(classes_dict.values())
    columns.append("total")
    columns.append("fname")

    df_to_geojson(output_geojson_file, df_result, columns)

    # Get max number per classe
    for class_name in columns:
        max_num = df_result[class_name].max()
        print(class_name, max_num)


if __name__ == "__main__":
    main()
