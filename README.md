# AI4Earth: AI Assisted Aerial Imagery Analysis for Wildlife Population and Human Conflict in Tanzania

Mitigating wildlife and human conflicts with AI, funded by Microsoft AI for Earth Innovation Grant and Global Wildlife Conservation.

## Labelling

Data labeling is done using [Computer Vision Annotation Tool (CVAT)](https://github.com/openvinotoolkit/cvat) tool.

![aiaia-training data creation](https://user-images.githubusercontent.com/14057932/106537594-a33f3f00-64c8-11eb-8cb0-8cb02c18b68b.png)
*Training dataset annotation workflow using CVAT. Aerial imagery was annotated by a group of volunteers in Tanzania. The annotators created 30 classes of labels that covered wildlife, livestock, and human activities. The final labeled data is tiled/chipped and converted into TFRecords format as machine learning ready data for the coming model training.*
### Aerial survey and the images

Aerial surveys of large mammals in Africa are normally done from light aircraft (Cessna 172/182/206 type) at an altitude above ground level (AGL) of 90-110m (300 to 350 feet).


There are two types of images that are available from aerial surveys for this project:

#### RSO

During the survey, Rear seat observers (RSO), window-mounted cameras are used to verify herd size and ID - they are triggered by the *human observer* when he/she sees a target of interest (all elephants, and larger herds of any species).
The aircaft flies straight lines (transects) back and forth over the target area.
  * These images are lower quality as they are taken through the Plexiglas.
  * Most images 24 megapixel (MP), taken at 35° down-angle from horizontal.
  * Very target-rich set of images which will produce many examples for training. This is because a human has *triggered* the camera on a target.
  * Arosund 20,000 images are currently available and being reviewed by the annotation lab.
  * GPS metadata are often not available (camera clocks not synchronized perfectly with GPS).


#### New aerial survey

The new system is for cameras on the wing struts which take constant images along flight paths.
  * These images have no intervening Plexiglas and are much higher quality.
  * Taken at 45° down-angle.
  * 24MP and optimized for image sharpness.
  * Very low rate of 'positives' - perhaps 2% of images will have any wildlife or livestock present.
  * Around 500,000 images are available.
  * GPS metadata present for all images.

  <img width="903" alt="PAS-aerial-survey" src="https://user-images.githubusercontent.com/14057932/106537752-f6b18d00-64c8-11eb-99f7-f42cbc457264.png">

## Training Data Generation
-  Build and access the container

```sh
cd AIAIA/util_data
docker-compose build
USERID=$(id -u) docker-compose run aiaia_tf bash
```

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

## AI-assisted Aerial Imagery Analysis (AIAIA)

At the end of the project, we present two AI-assisted systems: 1) an image classifier, AIAIA Classifier, that filters images containing objects of interest from tens of thousands of aerial images using automated camera systems and 2) a set of three object detection models, AIAIA Detectors, which locate, classify and count objects of interest within those images. The detected objects were assigned to image IDs that have their unique geolocation recorded during the aerial surveys. These geocoded detections were then used to generate maps of the distribution of wildlife, human activities, and livestock, with a visualisation of mapped proximity highlighting potential conflict areas. Explore the map here.

<img width="1131" alt="AIAIA-workflow" src="https://user-images.githubusercontent.com/14057932/106538113-b6064380-64c9-11eb-86f4-602bc8ab41dc.png">

### AIAIA Classifier

#### Building the training image

To build the training image, run the following command, replacing `VERSION` with an
appropriate value (e.g., `v2`):

```bash
cd aiaia_classifier/
export VERSION=v1
export PROJECT=bp-padang
docker build . -t gcr.io/${PROJECT}/aiaia-classifier:${VERSION}-xception-binary
```

* Uses a google deep learning base with tensorflow 2 and the correct CUDA
  version (10.0) pre-installed -- note that this container does not support tensorflow
  addons package.
* Corresponds to Dockerfile in repo.

If it's the first time you open the PROJECT before you push the image to GCR, it will ask you to 'Enable' the GCR API on the Cloud Console, e.g. 'https://console.cloud.google.com/apis/api/containerregistry.googleapis.com/overview?project=project-connect-289520'
And enable other [API resources](https://www.kubeflow.org/docs/gke/deploy/project-setup/), e.g., running:

```bash

gcloud services enable \                         
  compute.googleapis.com \
  container.googleapis.com \
  iam.googleapis.com \
  servicemanagement.googleapis.com \
  cloudresourcemanager.googleapis.com \
  ml.googleapis.com
```

Now, you can push the image to GCR using the following command:

```bash
docker push gcr.io/${PROJECT}/aiaia-classifier:${VERSION}-xception-binary
```

Make sure that the docker info in the `katib` file matches the most up-to date
version of the dockerfile.

#### Setup Kubeflow on GCP cluster

Assuming you have correctly specified values in your `.env` file as described
above, as well as installed the prerequisite tools (also described above), you
should be able to deploy a new cluster with the following command. (**NOTE:**
this will automatically detect and install the correct versions of `kubectl`
and `kfctl` for you, so there should be no need for you to do so manually.)

```bash
# TYPE is either 'standard' or 'highmem'
# Run ./deploy without args for details
./deploy --node-pool standard
```

**NOTE:** This will take about 15 minutes to complete!

#### Verify Resources

Once deployment is complete, run the following command to see all resources for
your cluster:

```
kubectl -n kubeflow get all
```

#### Single Experiment Run using TF-job

If you have already established the optimal combinations of hyper-parameters you
can create a tf-jobs yaml file. See yaml file in tf_jobs for example.

#### Start Experiment
Before you start the experiment, the tf_job yaml file will need to be updated, particularly the `tf_train_data_dir` and `tf_val_data_dir`.

```
kubectl create -f <path to tf job yaml file>
```

Check if the experiment is deployed properly, by running the following for logging:

```bash
stern -n kubeflow --since 10m --container tensorflow ".*"

```
#### Deploying hyperparameter optimization experiments

```
kubectl create -f <path to yaml file>

# see trials status
kubectl describe experiment <experiment_name> -n kubeflow

# see hyper-parameter combinations katib has generated
kubectl -n kubeflow describe suggestions

# useful to start looking at after trials have completed
kubectl -n kubeflow port-forward svc/katib-ui 8080:80
# then go to http://localhost:8080/katib/#/katib/hp_monitor for visualizations

# Delete running experiments like:
kubectl delete -f <path to yaml file>
```

#### Clean up

The following information will be printed out when the Kubeflow is successfully deployed.
You will need to add `--no-dry-run` to the following bash script to completely delete the cluster and resource once the ML training is finished.

```bash
# Delete cluster/resources once finished
./clean ${KF_DIR}
```
### Model evaluation

#### Running testing locally
Download TFRecords down from GCP either change the flags in aiaia_classifier/eval.py code.
Download the model checkpoint files from the best performing training step locally.
You can run the following command under `aiaia_classifier` directory:

```bash
python3 aiaia_classifier/eval.py \
       --tf_test_data_dir=dir_path_to_test_tfrecords/ \
       --country={country_name} \
       --tf_test_ckpt_path=dir_path_to_model_checkpoints/model.ckpt-6000 \
       --tf_test_results_dir=local_dir4_model_eval
```
*Note*: currently, model parameters under `eval.py` is hard coded for this aiaia classifier model.

#### Reading files from GCS for model testing

You can also point the model check point and test dataset that hosted on GCS. To do so, you will need to install packages:
- `pip3 install google-cloud-bigquery tenacity`, and
- log in to your GCP with `gcloud` and application authentication shows as follows:

```bash
# log in to the GCP so we can access to the files on GCS

gcloud init

gcloud auth application-default login

python3 aiaia_classifier/eval.py \
        --tf_test_data_dir='gs://dir_path_to_test_tfrecords/' \
        --countries={country_name} \
        --tf_test_ckpt_path='gs://dir_path_to_model_checkpoints/model_outputs/v1/model.ckpt-6000' \
        --tf_test_results_dir=local_dir4_model_eval
```

The above command will output three files:
- `preds.csv`;
- `test_stats.csv`; and
- `dist_fpr_tpr_{countries}.png` and `roc_{countries}.png` that shows model true and false positive rate and roc curve.


### AIAIA Detectors

#### Building the training image

To build the training image, run the following command, replacing `VERSION` with an
appropriate value (e.g., `v2`):

```bash
cd aiaia_detector/
export VERSION=v1
export PROJECT=bp-padang
docker build . -f Dockerfile-gpu -t gcr.io/${PROJECT}/aiaia:${VERSION}-tf1.15-gpu
docker build . -f Dockerfile-cpu -t gcr.io/${PROJECT}/aiaia:${VERSION}-tf1.15-cpu

```

#### Deploy TF jobs

If you have already established the optimal combinations of hyper-parameters you can create a tf-jobs yaml file. See yaml file in tf_jobs for example.

```bash
kubectl create -f <path to tf job yaml file>
```

Check if the experiment is deployed properly, by running the following for logging:

```bash
stern -n kubeflow --since 10m --container tensorflow ".*"
```

#### Delete TF experiment
To terminate the model experiment, run:

```bash
kubectl delete -f <path to tf job yaml file>
```

#### Watching model training with Tensorboard

After an object detection model is trained and model checkpoints saved in GCS. You can go through these few steps to visualize tensorboard:

##### Visualize tensorboard locally

```bash
gcloud init # to sign into your project with your email
gcloud auth application-default login

tensorboard --logdir='gs://aiaia_od/model_outputs_tf1/rcnn_resnet101_serengeti_wildlife_v3/'
```

##### Visualize tensorboard with Tensorboard Dev

```bash
gcloud init # to sign into your project with your email
gcloud auth application-default login

pip3 install -U tensorboard

## This will upload model files to tensorboard dev
tensorboard dev upload --logdir gs://aiaia_od/model_outputs_tf1/rcnn_resnet101_serengeti_wildlife_v3/

```

#### Evaluation

Evaluation was tested locally using the aiaia_tf docker image, which can be build with\

```
docker-compose build
```
from the root of this repository.

After downloading the frozen graph model files and TFRecords for the test datasets, you can run the evaluation script `run_all_eval.sh` from within the docker container. Make sure that paths in this script are correct for where you downloaded the files. In this case, the script is run from the aiaia_detector folder.

```
docker run -u 0 --rm --gpus all -v ${PWD}:/mnt/data -p 8888:8888 -it developmentseed/aiaia_tf:v1 bash run_all_eval.sh
```

This will save all outputs from the evaluation to three folders, `wildlife-outputs`, `livestock-outputs`, `human-activities-outputs`.

To run the container interactively

```
docker run -u 0 --rm --gpus all -v ${PWD}:/mnt/data -p 8888:8888 -it developmentseed/aiaia_tf:v1 bash
```

In the container, to view the options for evaluation run:

```
python evaluation.py --help
```


You can run a jupyter notebook from within the container with

```
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=./
```

Or use the Remote - Containers VSCode extension to run commands from within the container or use the VSCode debugger to inspect the evaluation script.


## Model inference

These scripts aim to run model inference with `frozen_inference_graph.pb`. It exporeted as "input_type" `image_tensor`, for more information on how to export a trained model, please see [this script](https://github.com/developmentseed/ai4earth-wildlife-conservation/blob/master/aiaia_detector_tf1/aiaia_detector/exporter.py). This is an alternative way of running model inference, you can run model inference with TFServing images we provided [here](../README.md). We faced a latency running with TFserving images.


### Running the scripts in local docker container

- Running the container and login in GCP

From: https://github.com/developmentseed/ai4earth-wildlife-conservation/blob/master/aiaia_detector_tf1/README.md#run-with-kubeflow

```bash
export VERSION=v1
export PROJECT=bp-padang
cd inference/frozen_graph/
docker run -u 0 --rm -v ${PWD}://mnt/data -it gcr.io/${PROJECT}/aiaia:${VERSION}-tf1.15-cpu bash


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

### Running on NVIDIA GPU on top of classifier result

```bash
docker pull devseeddeploy/od_frozen_graph_inference:v1-tf1.15-gpu

docker run --runtime=nvidia --gpus=all -u 0 --rm -v ${PWD}://mnt/data -it devseeddeploy/od_frozen_graph_inference:v1-tf1.15-gpu bash
```
#### Filter image chips by the threshold score

This is an optional if you have have result from an image classifier and only want to filter images that contains interested objects.

```bash
python3 sync_s3_keys_from_classifier.py \
                      --df_name=rr19_sl25_0.5.csv \
                      --threshold=0.75 \
                      --profile_name=xxxx \
                      --bucket=xxx \
                      --dest_dir=chips_sl25_rr19_75

```

#### Running inference

Under the NVIDIA docker container run the following script to execute inference with frozen graph.


```bash
python3 frozen_pred.py \
     --frozen_inference_graph=models/wildlife/frozen_inference_graph.pb \
     --images_path=chips_sl25_rr19_75/*/*.jpg \
     --threshold=0.5 \
     --batch_size=100
```