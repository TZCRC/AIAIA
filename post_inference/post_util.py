"""
Utils functions

"""
import json
import xml.etree.ElementTree as ET
from xml.etree import ElementTree
from xml.dom import minidom


def df_to_geojson(geojson_file, df, properties, lat="y", lon="x"):
    """Dataframe to gejson

    Args:
        geojson_file (String): path to geojsoon file
        df (Dataframe):
        properties (list): List of attributes to consider in the df
        lat (str, optional): Attribute name on Df referred to latitude. Defaults to "y". Defaults to "y".
        lon (str, optional): Attribute name on Df referred to longitude. Defaults to "x".
    """
    geojson = {"type": "FeatureCollection", "features": []}
    for _, row in df.iterrows():
        feature = {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Point", "coordinates": []},
        }
        feature["geometry"]["coordinates"] = [row[lon], row[lat]]
        for prop in properties:
            feature["properties"][prop] = row[prop]
        geojson["features"].append(feature)

    with open(geojson_file, "w") as f:
        f.write(json.dumps(geojson))


def filter_predictions(predictions, threshold):
    """Filter predictions for a threshold
    Args:
        predictions (Dict): predictions dictionary
        threshold (Int): threshold for filtering

    Returns:
        Dict: predictions dictionary
    """
    filtered_preds = []
    for pred in predictions:
        if len(pred["classes"]) > 0:
            filtered_boxes = []
            filtered_score = []
            filtered_classes = []
            for i, score in enumerate(pred["scores"]):
                if score > threshold:
                    filtered_boxes.append(pred["boxes"][i])
                    filtered_score.append(score)
                    filtered_classes.append(pred["classes"][i])

            filtered_preds.append(
                {
                    "image_id": pred["image_id"],
                    "category": pred["category"],
                    "boxes": filtered_boxes,
                    "scores": filtered_score,
                    "classes": filtered_classes,
                    "num": len(filtered_boxes),
                }
            )
    return filtered_preds


def add_classes_name(predictions):
    """Add class string name on the predictions

    Args:
        predictions (Dict)

    Returns:
        predictions (Dict)
    """
    class_names = {
        "wildlife": {
            1: "buffalo",
            2: "dark_coloured_large",
            3: "elephant",
            4: "giraffe",
            5: "hippopotamus",
            6: "light_coloured_large",
            7: "smaller_ungulates",
            8: "warthog",
            9: "zebra",
        },
        "livestock": {1: "cow", 2: "donkey", 3: "shoats"},
        "human_activities": {
            1: "boma",
            2: "building",
            3: "charcoal_mound",
            4: "charcoal_sack",
            5: "human",
        },
    }
    for pred in predictions:
        pred["classes_name"] = [
            class_names[pred["category"]][c] for c in pred["classes"]
        ]
    return predictions


def prettify(elem):
    """Format tree object

    Args:
        elem (ElementTree.Element): etree object

    Returns:
        reparsed xml
    """
    print(type(elem))

    rough_string = ElementTree.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")
