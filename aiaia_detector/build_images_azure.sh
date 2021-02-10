#!/bin/bash -e

export VERSION=v1
export PROJECT=bp-padang

full_image_name_cpu=aiaiatrain.azurecr.io/aiaia:${VERSION}-tf1.15-frozen-graph-cpu
full_image_name_gpu=aiaiatrain.azurecr.io/aiaia:${VERSION}-tf1.15-frozen-graph-gpu

cd "$(dirname "$0")"
docker build . -f Dockerfile-cpu -t $full_image_name_cpu &
docker build . -f Dockerfile-gpu -t $full_image_name_gpu

echo "pushing.. $full_image_name_cpu"
docker push $full_image_name_cpu
docker push $full_image_name_gpu