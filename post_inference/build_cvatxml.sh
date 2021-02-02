# !/usr/bin/env bash

export AWS_PROFILE=devseed
mkdir -p data/

##########################################################################################
## Get location data and merge in one file
##########################################################################################
## location
cp ../imagery/cormon2019.csv data/
cp ../imagery/mxj2019.csv data/
cp ../imagery/rr19.csv data/
cp ../imagery/sl25.csv data/

##########################################################################################
## Get prediction data
##########################################################################################
aws s3 cp s3://ds-satellite-projects/AIAIA_AI4Earth/inference/wildlife_inference_results.json data/
aws s3 cp s3://ds-satellite-projects/AIAIA_AI4Earth/inference/livestock_inference_results.json data/
aws s3 cp s3://ds-satellite-projects/AIAIA_AI4Earth/inference/human_activities_inference_results.json data/

jq --compact-output -s '[.[][]]' \
    data/wildlife_inference_results.json \
    data/livestock_inference_results.json \
    data/human_activities_inference_results.json >data/inference_results.json

# # ##########################################################################################
# # ## inferece to cvat xml format
# # ##########################################################################################

# cormon2019
python3 prediction2cvatxml.py \
    --csv_location_file=data/mxj2019.csv \
    --json_prediction_file=data/wildlife_inference_results.json  \
    --threshold=0.85 \
    --output_xml_file=data/cormon2019.xml
