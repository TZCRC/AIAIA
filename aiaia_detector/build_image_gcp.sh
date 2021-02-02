#!/bin/bash -e

export VERSION=v1
export PROJECT=bp-padang

full_image_name_cpu=gcr.io/${PROJECT}/aiaia:${VERSION}-tf1.15-cpu
full_image_name_gpu=gcr.io/${PROJECT}/aiaia:${VERSION}-tf1.15-gpu

cd "$(dirname "$0")"
docker build . -f Dockerfile-cpu -t $full_image_name_cpu &
docker build . -f Dockerfile-gpu -t $full_image_name_gpu

echo "pushing.. $full_image_name_cpu"
docker push $full_image_name_cpu
docker push $full_image_name_gpu

# # Output the strict image name (which contains the sha256 image digest)
# docker inspect --format="{{index .RepoDigests 0}}" "${$full_image_name_cpu}"
# docker inspect --format="{{index .RepoDigests 0}}" "${$full_image_name_gpu}"
