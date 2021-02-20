# AIAIA Classifier

### Building the training image

The image is already located on the Azure container registry `aiaiatrain`. If you need to build and push the training image again, run the following command, replacing v1.2 with an appropriate value.

```bash
az acr login --name aiaiatrain
docker build . -t v1.2-xception-binary-classifier:latest
docker push v1.2-xception-binary-classifier:latest
```

See root repo README for a guide on evaluating classifier results and the end of the aiaia_detector README.md for how to create a workspace and run experiments.