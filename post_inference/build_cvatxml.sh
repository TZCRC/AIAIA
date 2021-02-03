# !/usr/bin/env bash

export AWS_PROFILE=devseed
mkdir -p data/

##########################################################################################
## Get location data and merge in one file
##########################################################################################
## location
aws s3 sync s3://ds-data-projects/ai4earth-wildlife-conservation/csv/ data/

##########################################################################################
## Get prediction data
##########################################################################################
# aws s3 cp s3://ds-satellite-projects/AIAIA_AI4Earth/inference/wildlife_inference_results.json data/
# aws s3 cp s3://ds-satellite-projects/AIAIA_AI4Earth/inference/livestock_inference_results.json data/
# aws s3 cp s3://ds-satellite-projects/AIAIA_AI4Earth/inference/human_activities_inference_results.json data/

# jq --compact-output '.[].category="wildlife"' data/wildlife_inference_results.json > data/_wildlife.json
# jq --compact-output '.[].category="livestock"' data/livestock_inference_results.json > data/_livestock.json
# jq --compact-output '.[].category="human_activities"' data/human_activities_inference_results.json > data/_human_activities.json

# jq --compact-output -s '[.[][]]' \
#     data/_wildlife.json \
#     data/_livestock.json \
#     data/_human_activities.json >data/inference_results.json

# # ##########################################################################################
# # ## inferece to cvat xml format
# # ##########################################################################################

# mxj2019
python3 prediction2cvatxml.py \
    --csv_location_file=data/mxj2019.csv \
    --json_prediction_file=data/wildlife_inference_results.json \
    --threshold=0.85 \
    --images_path=data/mxj2019 \
    --output_xml_file=data/mxj2019.xml
