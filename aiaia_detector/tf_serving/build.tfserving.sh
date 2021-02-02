#!/bin/bash
# This is script is to build our own TFServing version considering some differents flag imports for Bazel server

####################################
# CPU version
####################################

USER=developmentseed
TAG=v1-1.15-cpu
TF_SERVING_VERSION_GIT_BRANCH="r1.15"
git clone --branch="${TF_SERVING_VERSION_GIT_BRANCH}" https://github.com/tensorflow/serving

TF_SERVING_BUILD_OPTIONS="--copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.1 --copt=-msse4.2"
cd serving &&
  docker build --pull -t $USER/tensorflow-serving-devel:$TAG \
    --build-arg TF_SERVING_VERSION_GIT_BRANCH="${TF_SERVING_VERSION_GIT_BRANCH}" \
    --build-arg TF_SERVING_BUILD_OPTIONS="${TF_SERVING_BUILD_OPTIONS}" \
    -f tensorflow_serving/tools/docker/Dockerfile.devel .
cd ../
cd serving &&
  docker build -t $USER/tensorflow-serving:$TAG \
    --build-arg TF_SERVING_BUILD_IMAGE=$USER/tensorflow-serving-devel:$TAG \
    -f tensorflow_serving/tools/docker/Dockerfile .

docker push $USER/tensorflow-serving:$TAG
####################################
# GPU version
####################################

USER=developmentseed
TAG=v1-1.15-gpu
TF_SERVING_VERSION_GIT_BRANCH="r1.15"
git clone --branch="${TF_SERVING_VERSION_GIT_BRANCH}" https://github.com/tensorflow/serving

TF_SERVING_BUILD_OPTIONS="--copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.1 --copt=-msse4.2"
cd serving &&
  docker build --pull -t $USER/tensorflow-serving-devel:$TAG \
    --build-arg TF_SERVING_VERSION_GIT_BRANCH="${TF_SERVING_VERSION_GIT_BRANCH}" \
    --build-arg TF_SERVING_BUILD_OPTIONS="${TF_SERVING_BUILD_OPTIONS}" \
    -f tensorflow_serving/tools/docker/Dockerfile.devel-gpu .
cd ../
cd serving &&
  docker build -t $USER/tensorflow-serving:$TAG \
    --build-arg TF_SERVING_BUILD_IMAGE=$USER/tensorflow-serving-devel:$TAG \
    -f tensorflow_serving/tools/docker/Dockerfile.gpu .

docker push $USER/tensorflow-serving:$TAG
