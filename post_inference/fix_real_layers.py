"""
Sum up all the values for aggregate layers
"""
import json
import pandas as pd
from post_util import df_to_geojson


def normalize_column(df, column):
    # copy the data
    df_min_max_scaled = df.copy()
    df_min_max_scaled[column + "_normalized"] = (
        df_min_max_scaled[column] - df_min_max_scaled[column].min()
    ) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())
    return df_min_max_scaled


wildlife_classes = [
    "buffalo",
    "dark_coloured_large",
    "elephant",
    "giraffe",
    "hippopotamus",
    "light_coloured_large",
    "smaller_ungulates",
    "warthog",
    "zebra",
]

livestock_classes = ["cow", "donkey", "shoats"]

human_activities_classes = [
    "boma",
    "building",
    "charcoal_mound",
    "charcoal_sack",
    "human",
]

with open("data/vis_data.geojson") as f:
    vis_data = json.load(f)["features"]
result = {}
for w in vis_data:
    group = w["properties"]["group"]
    count = w["properties"]["count"]
    if group is not None and count is not None:
        coords = ",".join(map(str, w["geometry"]["coordinates"]))
        if coords in result.keys():
            if group in result[coords].keys():
                result[coords][group] = result[coords][group] + count
        else:
            d = {}
            d["x"] = w["geometry"]["coordinates"][0]
            d["y"] = w["geometry"]["coordinates"][1]
            d[group] = count
            result[coords] = d


df = pd.DataFrame.from_dict(result.values())
df = df.fillna(0)
df["hippopotamus"] = 0
df["charcoal_sack"] = 0
df["charcoal_mound"] = 0
df["human"] = 0
# SUM values
df["wildlife"] = df[wildlife_classes].sum(axis=1)
df["livestock"] = df[livestock_classes].sum(axis=1)
df["human_activities"] = df[human_activities_classes].sum(axis=1)
df["total"] = df[
    wildlife_classes + livestock_classes + human_activities_classes
].sum(axis=1)

# Normalize column
df = normalize_column(df, "wildlife")
df = normalize_column(df, "livestock")
df = normalize_column(df, "human_activities")

# print(df)
df.to_csv("data/real_ai4earth_results.csv", index=False)


df_wildlife = df[
    wildlife_classes + ["x", "y", "wildlife", "wildlife_normalized"]
]
df_livestock = df[
    livestock_classes + ["x", "y", "livestock", "livestock_normalized"]
]
df_human_activities = df[
    human_activities_classes
    + ["x", "y", "human_activities", "human_activities_normalized"]
]
df_result = df[
    [
        "x",
        "y",
        "wildlife",
        "livestock",
        "human_activities",
        "total",
    ]
]

# Filter values that are greater than 0
df_wildlife.rename(columns={"wildlife": "total"}, inplace=True)
df_livestock.rename(columns={"livestock": "total"}, inplace=True)
df_human_activities.rename(columns={"human_activities": "total"}, inplace=True)
df_wildlife = df_wildlife[df_wildlife["total"] > 0]
df_livestock = df_livestock[df_livestock["total"] > 0]
df_human_activities = df_human_activities[df_human_activities["total"] > 0]
df_result = df_result[df_result["total"] > 0]

df_to_geojson(
    "data/real_wildlife_results.geojson",
    df_wildlife,
    wildlife_classes + ["total"],
)

df_to_geojson(
    "data/real_livestock_results.geojson",
    df_livestock,
    livestock_classes + ["total"],
)

df_to_geojson(
    "data/real_human_activities_results.geojson",
    df_human_activities,
    human_activities_classes + ["total"],
)

df_to_geojson(
    "data/real_ai4earth_results.geojson",
    df_result,
    ["wildlife", "livestock", "human_activities", "total"],
)

# Get max values
for cl in (
    wildlife_classes
    + livestock_classes
    + human_activities_classes
    + ["wildlife", "livestock", "human_activities", "total"]
):
    max_num = df[cl].max()
    print(cl, max_num)
