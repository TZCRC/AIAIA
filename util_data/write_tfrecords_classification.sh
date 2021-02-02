# !/usr/bin/env bash

export AWS_PROFILE=ai4e
aiaia_tf="docker run --rm -v ${PWD}:/mnt/data developmentseed/aiaia_tf:v1"

# #########################################################
# ### Download csv and tiles
# #########################################################
aws s3 sync s3://aisurvey/training_data202008/P400_v2/ data/csv/ --exclude="*" --include="*.csv"

FOLDERS=("TA25" "RR19" "RR17" "SL25")

for folder in ${FOLDERS[*]}; do
    aws s3 sync s3://aisurvey/training_data202008/P400_v2/${folder}_tiles/ data/P400_v2/${folder}_tiles/
done

# #########################################################
# ### Creating tfrecords
# #########################################################
# Note: Make sure the container has tensorlflow >= 2.x
$aiaia_tf python tf_records_creation_classification.py \
    --tile_path=data/P400_v2/ \
    --csv_files=data/csv/*_class_id.csv \
    --output_dir=data/classification_training_tfrecords/ \
    --output_csv=data/csv/classification_training_tfrecords.csv

# #########################################################
# ### Upload gsp
# #########################################################

gsutil rsync -r data/classification_training_tfrecords/ gs://aiaia_od/training_data_aiaia_p400/classification_training_tfrecords/
aws s3 cp data/csv/classification_training_tfrecords.csv s3://aisurvey/training_data202008/P400_v2/
gsutil rsync -r data/csv/ gs://aiaia_od/csv_files/
