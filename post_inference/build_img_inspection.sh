# !/usr/bin/env bash

mkdir -p data/

##########################################################################################
## Get data results aand merge in one json file
##########################################################################################
export AWS_PROFILE=devseed

# #### Wildlife
aws s3 sync  s3://ds-satellite-projects/AIAIA_AI4Earth/inference/wildlife_results/ data/wildlife_results/
aws s3 sync  s3://ds-satellite-projects/AIAIA_AI4Earth/inference/rr19_sl25_wildlife_results/ data/rr19_sl25_wildlife_results/

jq --compact-output -s '[.[][]]' data/wildlife_results/*.json > data/cormon2019_mxj2019_wildlife_results.json
jq --compact-output -s '[.[][]]' data/rr19_sl25_wildlife_results/*.json > data/rr19_sl25_wildlife_results.json
jq --compact-output -s '[.[][]]' data/cormon2019_mxj2019_wildlife_results.json data/rr19_sl25_wildlife_results.json > data/wildlife_inference_results.json

# #### Livestock
aws s3 sync  s3://ds-satellite-projects/AIAIA_AI4Earth/inference/livestock_results/ data/livestock_results/
aws s3 sync  s3://ds-satellite-projects/AIAIA_AI4Earth/inference/rr19_sl25_livestock_results/ data/rr19_sl25_livestock_results/

jq --compact-output -s '[.[][]]' data/livestock_results/*.json > data/cormon2019_mxj2019_livestock_results.json
jq --compact-output -s '[.[][]]' data/rr19_sl25_livestock_results/*.json > data/rr19_sl25_livestock_results.json
jq --compact-output -s '[.[][]]' data/cormon2019_mxj2019_livestock_results.json data/rr19_sl25_livestock_results.json > data/livestock_inference_results.json


# #### Livestock
aws s3 sync  s3://ds-satellite-projects/AIAIA_AI4Earth/inference/human_activities_results/ data/human_activities_results/
aws s3 sync  s3://ds-satellite-projects/AIAIA_AI4Earth/inference/rr19_sl25_human_activities_results/ data/rr19_sl25_human_activities_results/

jq --compact-output -s '[.[][]]' data/human_activities_results/*.json > data/cormon2019_mxj2019_human_activities_results.json
jq --compact-output -s '[.[][]]' data/rr19_sl25_human_activities_results/*.json > data/rr19_sl25_human_activities_results.json
jq --compact-output -s '[.[][]]' data/cormon2019_mxj2019_human_activities_results.json data/rr19_sl25_human_activities_results.json > data/human_activities_inference_results.json

# ##### Upload to s3
aws s3 cp data/cormon2019_mxj2019_wildlife_results.json s3://ds-satellite-projects/AIAIA_AI4Earth/inference/
aws s3 cp data/cormon2019_mxj2019_livestock_results.json s3://ds-satellite-projects/AIAIA_AI4Earth/inference/
aws s3 cp data/cormon2019_mxj2019_human_activities_results.json s3://ds-satellite-projects/AIAIA_AI4Earth/inference/

aws s3 cp data/rr19_sl25_wildlife_results.json s3://ds-satellite-projects/AIAIA_AI4Earth/inference/
aws s3 cp data/rr19_sl25_livestock_results.json s3://ds-satellite-projects/AIAIA_AI4Earth/inference/
aws s3 cp data/rr19_sl25_human_activities_results.json s3://ds-satellite-projects/AIAIA_AI4Earth/inference/

aws s3 cp data/wildlife_inference_results.json s3://ds-satellite-projects/AIAIA_AI4Earth/inference/
aws s3 cp data/livestock_inference_results.json s3://ds-satellite-projects/AIAIA_AI4Earth/inference/
aws s3 cp data/human_activities_inference_results.json s3://ds-satellite-projects/AIAIA_AI4Earth/inference/

# ##########################################################################################
# # Build container for running the scripts
# ##########################################################################################
# Run it in a docker container
# docker run --rm -v ${PWD}:/mnt/data -it developmentseed/aiaia_tf:v1

# wildlife
python draw_bboxes_over_chip.py \
        --json_file=data/cormon2019_mxj2019_wildlife_results.json \
        --aws_bucket=aisurvey \
        --random_sample=0.2 \
        --out_chip_dir=data/wildlife_inspect


python draw_bboxes_over_chip.py \
        --json_file=data/rr19_sl25_wildlife_results.json \
        --aws_bucket=aiaia-inference \
        --random_sample=1 \
        --out_chip_dir=data/wildlife_inspect

# livestock
python draw_bboxes_over_chip.py \
        --json_file=data/cormon2019_mxj2019_livestock_results.json \
        --category=livestock \
        --aws_bucket=aisurvey \
        --random_sample=0.2 \
        --out_chip_dir=data/livestock_inspect


python draw_bboxes_over_chip.py \
        --json_file=data/rr19_sl25_livestock_results.json \
        --category=livestock \
        --aws_bucket=aiaia-inference \
        --random_sample=1 \
        --out_chip_dir=data/livestock_inspect


# human_activities
python draw_bboxes_over_chip.py \
        --json_file=data/cormon2019_mxj2019_human_activities_results.json \
        --category=human_activities \
        --aws_bucket=aisurvey \
        --random_sample=0.2 \
        --out_chip_dir=data/human_activities_inspect


python draw_bboxes_over_chip.py \
        --json_file=data/rr19_sl25_human_activities_results.json \
        --category=human_activities \
        --aws_bucket=aiaia-inference \
        --random_sample=1 \
        --out_chip_dir=data/human_activities_inspect

# ##########################################################################################
# # Upload images to s3
# ##########################################################################################

aws s3 sync data/wildlife_inspect/ s3://ds-satellite-projects/AIAIA_AI4Earth/inference/wildlife_results_inspection/
aws s3 sync data/livestock_inspect/ s3://ds-satellite-projects/AIAIA_AI4Earth/inference/livestock_results_inspection/
aws s3 sync data/human_activities_inspect/ s3://ds-satellite-projects/AIAIA_AI4Earth/inference/human_activities_results_inspection/
