# Frozen prediction

These scripts aim to run model inference with `frozen_inference_graph.pb`. It exporeted as "input_type" `image_tensor`, for more information on how to export a trained model, please see [this script](https://github.com/developmentseed/ai4earth-wildlife-conservation/blob/master/aiaia_detector_tf1/aiaia_detector/exporter.py). This is an alternative way of running model inference, you can run model inference with TFServing images we provided [here](../README.md). We faced a latency running with TFserving images.


## Running the scripts in local docker container

- Running the container and login in GCP

From: https://github.com/developmentseed/ai4earth-wildlife-conservation/blob/master/aiaia_detector_tf1/README.md#run-with-kubeflow

```bash
export VERSION=v1
export PROJECT=bp-padang
cd inference/frozen_graph/
docker run -u 0 --rm -v ${PWD}://mnt/data -it gcr.io/${PROJECT}/aiaia:${VERSION}-tf1.15-cpu bash

# Once in the container
gcloud init
gcloud auth application-default login

```

- Getting required files

```bash
# create folder
mkdir -p models/wildlife
mkdir -p data/chips/

# Sync files from bucket files to local
gsutil cp gs://aiaia_od/export_outputs_tf1/ryan_sprint_rcnn_resnet101_serengeti_wildlife_v3_tfs/frozen_inference_graph.pb models/wildlife/
gsutil rsync gs://aiaia_od/chips data/chips/

# copy pbtxt file for wildlife
gsutil cp gs://aiaia_od/model_configs/labels/wildlife.pbtxt data/
```

- Executing prediction

```bash
# in case some pip modules is missing in the container.
pip install numpyencoder click joblib tqdm

python frozen_pred.py \
    --frozen_inference_graph=models/wildlife/frozen_inference_graph.pb \
    --images_path=data/chips/*.jpg \
    --threshold=0.5
```

The output will be store in the file: ` data/result.json`

## Running on NVIDIA GPU on top of classifier result

```bash
docker pull devseeddeploy/od_frozen_graph_inference:v1-tf1.15-gpu

docker run --runtime=nvidia --gpus=all -u 0 --rm -v ${PWD}://mnt/data -it devseeddeploy/od_frozen_graph_inference:v1-tf1.15-gpu bash
```
### Filter image chips by the threshold score

This is an optional if you have have result from an image classifier and only want to filter images that contains interested objects.

```bash
python3 sync_s3_keys_from_classifier.py \
                      --df_name=rr19_sl25_0.5.csv \
                      --threshold=0.75 \
                      --profile_name=xxxx \
                      --bucket=xxx \
                      --dest_dir=chips_sl25_rr19_75

```

### Running inference

Under the NVIDIA docker container run the following script to execute inference with frozen graph.


```bash
python3 frozen_pred.py \
     --frozen_inference_graph=models/wildlife/frozen_inference_graph.pb \
     --images_path=chips_sl25_rr19_75/*/*.jpg \
     --threshold=0.5 \
     --batch_size=100
```
