# !/usr/bin/env bash

export AWS_PROFILE=devseed
mkdir -p data/
geokit="docker run -v ${PWD}:/mnt/data developmentseed/geokit:latest"

##########################################################################################
## Get location data and merge in one file
##########################################################################################
## location
cp ../imagery/cormon2019.csv data/
cp ../imagery/mxj2019.csv data/
cp ../imagery/rr19.csv data/
cp ../imagery/sl25.csv data/

$geokit csvcut -c sourcefile,fname,x,y data/cormon2019.csv >data/cormon2019_simple.csv
$geokit csvcut -c sourcefile,fname,x,y data/mxj2019.csv >data/mxj2019_simple.csv
$geokit csvcut -c sourcefile,fname,x,y data/rr19.csv >data/rr19_simple.csv
$geokit csvcut -c sourcefile,fname,x,y data/sl25.csv >data/sl25_simple.csv

cat data/mxj2019_simple.csv >data/ai4earth_locations.csv
awk 'FNR > 1' data/cormon2019_simple.csv \
    data/rr19_simple.csv \
    data/sl25_simple.csv >>data/ai4earth_locations.csv

##########################################################################################
## Get prediction data
##########################################################################################
aws s3 cp s3://ds-satellite-projects/AIAIA_AI4Earth/inference/wildlife_inference_results.json data/
aws s3 cp s3://ds-satellite-projects/AIAIA_AI4Earth/inference/livestock_inference_results.json data/
aws s3 cp s3://ds-satellite-projects/AIAIA_AI4Earth/inference/human_activities_inference_results.json data/
aws s3 cp s3://ds-data-projects/ai4earth/vis_data.geojson data/

# # ##########################################################################################
# # ## Add count column and convert to geojson
# # ##########################################################################################

echo " Matching prediction data with geolocation for Wildlife"
python3 match_pred_images.py \
    --csv_location_file=data/ai4earth_locations.csv \
    --json_prediction_file=data/wildlife_inference_results.json \
    --category=wildlife \
    --threshold=0.85 \
    --output_csv_file=data/wildlife_inference_results.csv \
    --output_geojson_file=data/inference_wildlife_results.geojson

echo " Matching prediction data with geolocation for Livestock"
python3 match_pred_images.py \
    --csv_location_file=data/ai4earth_locations.csv \
    --json_prediction_file=data/livestock_inference_results.json \
    --category=livestock \
    --threshold=0.85 \
    --output_csv_file=data/livestock_inference_results.csv \
    --output_geojson_file=data/inference_livestock_results.geojson

echo " Matching prediction data with geolocation for Human_activities"
python3 match_pred_images.py \
    --csv_location_file=data/ai4earth_locations.csv \
    --json_prediction_file=data/human_activities_inference_results.json \
    --category=human_activities \
    --threshold=0.85 \
    --output_csv_file=data/human_activities_inference_results.csv \
    --output_geojson_file=data/inference_human_activities_results.geojson

# # # ##########################################################################################
# # # ## Generate inference aggregate layers
# # # ##########################################################################################

# python3 fix_inference_layers.py

# # # ##########################################################################################
# # # ## Generate real aggregate layers
# # # ##########################################################################################

python3 fix_real_layers.py
# outputs
# "data/real_wildlife_results.geojson",
# "data/real_livestock_results.geojson",
# "data/real_human_activities_results.geojson",
# "data/real_ai4earth_results.geojson",

$geokit geokit buffer data/real_wildlife_results.geojson \
    --unit=kilometers --radius=10 --prop=wildlife_normalized >data/real_wildlife_results_buffer.geojson

$geokit geokit buffer data/real_livestock_results.geojson \
    --unit=kilometers --radius=10 --prop=livestock_normalized >data/real_livestock_results_buffer.geojson

$geokit geokit buffer data/real_human_activities_results.geojson \
    --unit=kilometers --radius=10 --prop=human_activities_normalized >data/real_human_activities_results_buffer.geojson

# # ##########################################################################################
# # ## Build mbtiles
# # ##########################################################################################

rm data/*.mbtiles
tippecanoe -Z0 -z5 -o data/ai4earth_inference.mbtiles \
    -L'{"layer": "wildlife", "file": "data/inference_wildlife_results.geojson"}' \
    -L'{"layer": "livestock", "file": "data/inference_livestock_results.geojson"}' \
    -L'{"layer": "human_activities", "file": "data/inference_human_activities_results.geojson"}' \
    -L'{"layer": "aggregate", "file": "data/inference_ai4earth_results.geojson"}'

tippecanoe -Z0 -z5 -o data/ai4earth_real.mbtiles \
    -L'{"layer": "wildlife", "file": "data/real_wildlife_results.geojson"}' \
    -L'{"layer": "livestock", "file": "data/real_livestock_results.geojson"}' \
    -L'{"layer": "human_activities", "file": "data/real_human_activities_results.geojson"}' \
    -L'{"layer": "aggregate", "file": "data/real_ai4earth_results.geojson"}'

# # # # # # ##########################################################################################
# # # # # # ## Upload to mapbox
# # # # # # ##########################################################################################
mapbox upload aiaia.ai4earth_inference data/ai4earth_inference.mbtiles
mapbox upload aiaia.ai4earth_real_v1 data/ai4earth_real.mbtiles
