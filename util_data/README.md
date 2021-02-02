# Utilities for ml training data generation

-  Build and access the container
  
```sh
cd ai4earth-wildlife-conservation/
docker-compose build
USERID=$(id -u) docker-compose run aiaia_tf bash
```

Once eni the docker coontainer ,run the followinf comands

Once in the docker container ,run the following commands 👇

## TFrecords creation

- Adding label_id, category and group in the CSV files
  
    From: https://docs.google.com/spreadsheets/d/1zWjgRcFwZh_OfpE8bUQoBBC3Imf8Ixua4eYBurjMUG4/edit#gid=0

```sh
python3 add_class_id.py \
    --csv=TA25_train_sliced_image_nbboxes.csv \
    --csv_output=TA25_train_sliced_image_nbboxes_class_id.csv
```

Files were stored at: `s3://aisurvey/training_data202008/P1000/*_train_sliced_image_nbboxes_class_id.csv`


- Creating Tfrecords for object detection

```sh
python3 tf_records_creation.py \
    --tile_path=aisurvey/training_data202008/P1000/SL25_tiles \
    --csv=aisurvey/training_data202008/P1000/SL25_train_sliced_image_nbboxes_class_id.csv \
    --csv_class_map=aisurvey/class_map/class_map.csv \
    --output_dir=/aisurvey/training_data202008/P1000/SL25_tfrecord \
    --width=1000 \
    --height=1000
```

or execute the bash file: 

```
./write_tfrecords.sh
```

## PBTXT creation

```
python3 write_training_pbtxt.py \
    --csv aisurvey/class_map/class_map.csv \
    --out_dir=aisurvey/training_data202008/P1000/pbtext/
```

outputs at: `s3://aisurvey/training_data202008/P1000/pbtxt`


- Creating Tfrecords for classification

```sh
python3 tf_records_creation_classification.py \
        --tile_path=data/P400_v2/ \
        --csv_files=data/csv/*_class_id.csv \
        --output_dir=data/classification_training_tfrecords/ \
        --output_csv=data/csv/classification_training_tfrecords.csv
```

or execute the bash file: 

```
./write_tfrecords_classification.sh

```