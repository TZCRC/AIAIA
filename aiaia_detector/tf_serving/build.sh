#!/bin/bash -e

############################################################
## Download exported trained models
############################################################
# log into and authorize your GCP
#gcloud init
#gcloud auth application-default login

mkdir -p models/human_activities/001/
mkdir -p models/livestock/001/
mkdir -p models/wildlife/001/
gsutil rsync -r gs://aiaia_od/export_outputs_tf1/rcnn_resnet101_serengeti_human_activities_v1_tfs_v2/saved_model/ models/human_activities/001/
gsutil rsync -r gs://aiaia_od/export_outputs_tf1/rcnn_resnet101_serengeti_livestock_v1_50k_tfs_v2/saved_model/ models/livestock/001/
gsutil rsync -r gs://aiaia_od/export_outputs_tf1/rcnn_resnet101_serengeti_wildlife_v3_tfs_v2/saved_model/ models/wildlife/001/

modules=(
    human_activities
    livestock
    wildlife
)
# v1.1 was built with TF1.15
# v1.2 was built with TF2.3.0
version=v1.2

for module in "${modules[@]}"; do
    ############################################################
    # Building CPU version TFServing image for local testing
    ############################################################
    echo "building.... developmentseed/aiaia_fastrcnn:${version}_${module}-cpu"
    docker run -d --name serving_base_${module} tensorflow/serving:2.3.0
    docker cp models/${module} serving_base_${module}:/models/${module}
    docker commit --change "ENV MODEL_NAME ${module}" serving_base_${module} devseeddeploy/aiaia_fastrcnn:${version}_${module}-cpu
    docker kill serving_base_${module}
    docker container prune
    docker push devseeddeploy/aiaia_fastrcnn:${version}_${module}-cpu

    echo docker run -p 8501:8501 -it devseeddeploy/aiaia_fastrcnn:${version}_${module}-cpu
    echo http://localhost:8501/${version}/models/${module}

    # ############################################################
    # # Building GPU version TFServing image for Chip n Scale
    # ############################################################
    echo "building.... developmentseed/aiaia_fastrcnn:${version}_${module}-gpu"
    docker run -d --name serving_base_${module}_gpu tensorflow/serving:2.3.0-gpu
    docker cp models/${module} serving_base_${module}_gpu:/models/${module}
    docker commit --change "ENV MODEL_NAME ${module}" serving_base_${module}_gpu devseeddeploy/aiaia_fastrcnn:${version}_${module}-gpu
    docker kill serving_base_${module}_gpu
    docker container prune
    docker push devseeddeploy/aiaia_fastrcnn:${version}_${module}-gpu
done
