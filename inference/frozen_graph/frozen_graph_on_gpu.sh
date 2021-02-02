#!/bin/bash -e

mkdir -p models/wildlife/
gsutil rsync -r gs://aiaia_od/export_outputs_tf1/ryan_sprint_rcnn_resnet101_serengeti_wildlife_v3_tfs/frozen_inference_graph.pb models/wildlife/

gsutil cp gs://aiaia_od/model_configs/labels/wildlife.pbtxt models/wildlife/
gsutil cp gs://aiaia_od/classification_inference/results_classification_0.5.csv .
# step 1, sync image from classifier result here:

python sync_s3_keys_from_classifier.py \
                      --df_name=results_classification_0.5.csv \
                      --threshold=0.75 \
                      --profile_name=aisurvey-nana \
                      --bucket=aisurvey \
                      --dest_dir=chips

echo "Image chips synced to chips at thresold 0.75, starting the inference on the frozen inference graph"
python frozen_pred.py \
     --frozen_inference_graph=models/wildlife/frozen_inference_graph.pb \
     --images_path=chips/*/*.jpg \
     --threshold=0.5 \
     --batch_size=10
