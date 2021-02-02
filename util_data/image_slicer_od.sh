# !/usr/bin/env bash

export AWS_PROFILE=ai4e

######################################
# Build the container
######################################
# # cd ai4earth-wildlife-conservation
# # docker-compose build
aiaia_tf="docker run --rm -v ${PWD}:/mnt/data developmentseed/aiaia_tf:v1"
mkdir -p data/

######################################
# cormon2019
######################################
# Create a CSV file for all cormon2019
echo "image_id,label,bbox,category" >cormon2019.csv
bucket_folder=s3://aisurvey/cormon2019/
for s3File in $(aws s3 ls $bucket_folder | awk '{print $4}'); do
  echo "data/cormon2019/${s3File},topi,\"[100, 100, 100, 100]\",wildlife" >>cormon2019.csv
done

# download files
aws s3 sync s3://aisurvey/cormon2019/ data/cormon2019/

# slice the images
$aiaia_tf python image_slicer_od.py \
  --in_csv=cormon2019.csv \
  --dimention=400 \
  --out_csv=data/cormon2019_nbboxes.csv \
  --out_tile_dir=data/cormon2019_chips/

# upload chips to s3
aws s3 sync data/cormon2019_chips/ s3://aisurvey/cormon2019_chips/

# ######################################
# # MXJ2019_full
# ######################################
# Create a CSV file for all MXJ2019_full
echo "image_id,label,bbox,category" >MXJ2019_full.csv
bucket_folder=s3://aisurvey/MXJ2019_full/
for s3File in $(aws s3 ls $bucket_folder | awk '{print $4}'); do
  echo "data/MXJ2019_full/${s3File},topi,\"[100, 100, 100, 100]\",wildlife" >>MXJ2019_full.csv
done

# download files
aws s3 sync s3://aisurvey/MXJ2019_full/ data/MXJ2019_full/

# slice the images
$aiaia_tf python3 image_slicer_od.py \
  --in_csv=MXJ2019_full.csv \
  --dimention=400 \
  --out_csv=data/mxj2019_full_nbboxes.csv \
  --out_tile_dir=data/mxj2019_full_chips/

# upload chips to s3
aws s3 sync data/mxj2019_full_chips/ s3://aisurvey/mxj2019_full_chips/

######################################
# SL25
######################################
# download files
aws s3 sync s3://aisurvey/SL25/ data/SL25/

# Create a CSV file for all SL25
echo "image_id,label,bbox,category" >SL25.csv
for img in data/SL25/*/*.jpg; do
  echo "$img,topi,\"[100, 100, 100, 100]\",wildlife" >>SL25.csv
done

# # slice the images
$aiaia_tf python3 image_slicer_od.py \
  --in_csv=SL25.csv \
  --dimention=400 \
  --out_csv=data/SL25_nbboxes.csv \
  --out_tile_dir=data/SL25_chips/

# upload chips to devseed bucket
aws s3 sync data/SL25_chips/ s3://aiaia-inference/SL25_chips/
aws s3 ls s3://aiaia-inference/SL25_chips/ | awk '{print "SL25_chips/"$4}' > SL25_chips.txt
aws s3 cp SL25_chips.txt s3://aiaia-inference/SL25_chips.txt

######################################
# RR19
######################################
# download files
aws s3 sync s3://aisurvey/RR19/ data/RR19/

# Create a CSV file for all RR19
echo "image_id,label,bbox,category" >RR19.csv
for img in data/RR19/*/*.jpg; do
  echo "$img,topi,\"[100, 100, 100, 100]\",wildlife" >>RR19.csv
done

# # slice the images
$aiaia_tf python3 image_slicer_od.py \
  --in_csv=RR19.csv \
  --dimention=400 \
  --out_csv=data/RR19_nbboxes.csv \
  --out_tile_dir=data/RR19_chips/

# upload chips to devseed bucket
aws s3 sync data/RR19_chips/ s3://aiaia-inference/RR19_chips/
aws s3 ls s3://aiaia-inference/RR19_chips/ | awk '{print "RR19_chips/"$4}' > RR19_chips.txt
aws s3 cp RR19_chips.txt s3://aiaia-inference/
