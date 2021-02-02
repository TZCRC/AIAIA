# !/usr/bin/env bash

export AWS_PROFILE=ai4e
aiaia_tf="docker run --rm -v ${PWD}:/mnt/data developmentseed/aiaia_tf:v1"

#########################################################
### Download required files
#########################################################

aws s3 sync s3://aisurvey/AIAIA_cvatxml/TA25 data/xml/TA25
aws s3 sync s3://aisurvey/AIAIA_cvatxml/SL25 data/xml/SL25
aws s3 sync s3://aisurvey/AIAIA_cvatxml/RR19 data/xml/RR19
aws s3 sync s3://aisurvey/AIAIA_cvatxml/RR17 data/xml/RR17

#########################################################
### XMl to csv
#########################################################
mkdir -p data/csv/

FOLDERS=("TA25" "RR19" "RR17" "SL25")

for folder in ${FOLDERS[*]}; do
    $aiaia_tf python xmls_to_df.py \
        --xml_path=data/xml/$folder/ \
        --csv_out=data/csv/$folder.csv
done

#########################################################
### Download images
#########################################################

aws s3 sync s3://aisurvey/RR17/ CVAT/RR17/
aws s3 sync s3://aisurvey/RR19/ CVAT/RR19/
aws s3 sync s3://aisurvey/SL25/ CVAT/SL25/
aws s3 sync s3://aisurvey/TA25/_data/RKE/TA25-RKE-20191202A/TA25-RKE-20191202A_L/ TA25/_data/RKE/TA25-RKE-20191202A/TA25-RKE-20191202A_L/

#########################################################
### Create image chips
#########################################################

for folder in ${FOLDERS[*]}; do
    $aiaia_tf python image_slicer_od.py \
        --in_csv=data/csv/${folder}.csv \
        --dimention=400 \
        --out_csv=data/csv/${folder}_train_sliced_image_nbboxes.csv \
        --out_tile_dir=data/chips/P400/${folder}_tiles/
done

#########################################################
### Add class Id
#########################################################

for folder in ${FOLDERS[*]}; do
    $aiaia_tf python add_class_id.py \
        --csv=data/csv/${folder}_train_sliced_image_nbboxes.csv \
        --csv_output=data/csv/${folder}_class_id.csv
done

#########################################################
### Draw bboxes over image chips
#########################################################

for folder in ${FOLDERS[*]}; do
    $aiaia_tf python draw_bboxes_over_image_chips.py \
        --csv=data/csv/${folder}_train_sliced_image_nbboxes.csv \
        --chip_dir=data/chips/P400/${folder}_tiles/ \
        --out_chip_dir=data/chips/P400/${folder}_tiles_inspect/
done

#########################################################
###  Write tfrecords
#########################################################

for folder in ${FOLDERS[*]}; do
    $aiaia_tf python tf_records_creation.py \
        --tile_path=data/chips/P400/${folder}_tiles/ \
        --csv=data/csv/${folder}_class_id.csv \
        --output_dir=data/chips/P400/${folder}_tfrecord \
        --width=400 \
        --height=400 \
        --split_in_chunks=True
done

#########################################################
###  Tfrecords to images
#########################################################

for folder in ${FOLDERS[*]}; do
    $aiaia_tf python tfrecord2images.py \
        --tfrecords_path=data/chips/P400/${folder}_tfrecord/*.tfrecords \
        --output_dir=data/chips/P400/${folder}_tfrecord_images
done


#########################################################
###  Upload results to s3 and GCP
#########################################################

aws s3 sync data/csv/ s3://aisurvey/training_data202008/P400_v2/
aws s3 sync data/chips/P400/ s3://aisurvey/training_data202008/P400_v2/

gsutil rsync -r data/chips/P400/RR17_tfrecord/ gs://aiaia_od/training_data_aiaia_p400/RR17_tfrecord/
gsutil rsync -r data/chips/P400/RR19_tfrecord/ gs://aiaia_od/training_data_aiaia_p400/RR19_tfrecord/
gsutil rsync -r data/chips/P400/SL25_tfrecord/ gs://aiaia_od/training_data_aiaia_p400/SL25_tfrecord/
gsutil rsync -r data/chips/P400/TA25_tfrecord/ gs://aiaia_od/training_data_aiaia_p400/TA25_tfrecord/
