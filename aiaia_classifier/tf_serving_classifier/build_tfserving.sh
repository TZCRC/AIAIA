#!/bin/bash -e
tf_version=2.3.0
model=xception_classifier
version=v1.0
mkdir -p models/${model}/001/
gsutil rsync -r gs://aiaia_od/classification_model_outputs/abc/export/abc/1610779099/ models/${model}/001/


############################################################
# Building CPU version TFServing image for local testing
############################################################
echo "building.... devseeddeploy/aiaia_classifier:${version}-cpu"
docker run -d --name serving_base_${model} tensorflow/serving:${tf_version}
docker cp models/${model} serving_base_${model}:/models/${model}
docker commit --change "ENV MODEL_NAME ${model}" serving_base_${model} devseeddeploy/aiaia_classifier:${version}-cpu
docker kill serving_base_${model}
docker container prune
docker push devseeddeploy/aiaia_classifier:${version}-cpu

echo docker run -p 8501:8501 -it devseeddeploy/aiaia_classifier:${version}-cpu
echo http://localhost:8501/${version}/models/${model}

# ############################################################
# # Building GPU version TFServing image for Chip n Scale
# ############################################################
echo "building.... devseeddeploy/aiaia_classifier:${version}-gpu"
docker run -d --name serving_base_${model}_gpu tensorflow/serving:${tf_version}-gpu
docker cp models/${model} serving_base_${model}_gpu:/models/${model}
docker commit --change "ENV MODEL_NAME ${model}" serving_base_${model}_gpu devseeddeploy/aiaia_classifier:${version}-gpu
docker kill serving_base_${model}_gpu
docker container prune
docker push devseeddeploy/aiaia_classifier:${version}-gpu
