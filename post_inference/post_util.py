"""
Utils functions

"""
import json
import xml.etree.ElementTree as ET
from xml.etree import ElementTree
from xml.dom import minidom


def df_to_geojson(geojson_file, df, properties, lat="y", lon="x"):
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
                    "boxes": filtered_boxes,
                    "scores": filtered_score,
                    "classes": filtered_classes,
                    "num": len(filtered_boxes),
                }
            )
    return filtered_preds


def prettify(elem):
    rough_string = ElementTree.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")
