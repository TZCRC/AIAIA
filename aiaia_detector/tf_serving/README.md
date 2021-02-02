# TFServing creation

Make sure that you have access to the GCP bucket `gs://aiaia_od`

```bash
    ./build.sh
```

# Check input format for TFServing

```bash
    USERID=$(id -u) docker-compose run aiaia_tf bash
    whereis saved_model_cli.py
    # It will print something like : /usr/local/bin/saved_model_cli, and then run 👇
    /usr/local/bin/saved_model_cli  show --dir /ai4earth-wildlife-conservation/aiaia_detector_tf1/tf_serving/human_activities/001/ --all
```

Output section:

```js
signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_STRING
        shape: (-1)
        name: encoded_image_string_tensor:0
```

So the input format will be 👇:

```js
payload = {"instances": [{"inputs": img}]}
```

# Testing TFServing

### Running up jupyter notebook

```bash
    cd ai4earth-wildlife-conservation/
    docker-compose build
    docker-compose up
```

Open the notebook and go to `aiaia_detector_tf1/tf_serving/`

### Running up TFServing

TFserving containers availables:

```bash
    developmentseed/aiaia_wildlife:v1
    developmentseed/aiaia_livestock:v1
    developmentseed/aiaia_human_activities:v1
```

- Execute a tfserving

```bash
    docker run \
    -p 8501:8501 \
    --network ai4earth-wildlife-conservation_default \
    -it developmentseed/aiaia_wildlife:v1-cpu
```

Check at: http://localhost:8501/v1/models/wildlife
