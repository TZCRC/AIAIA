# Running chip scale for AIAIA model inferences

```bash
git clone git@github.com:developmentseed/ai4earth-wildlife-conservation.git
git clone git@github.com:developmentseed/chip-n-scale-queue-arranger.git
```


## Classifier inference

To run classifier inference with Chip n Scale, please rename  `handler_classification.py` to be `handler.py`, and copy and replace  the following files:
    - `ai4earth-wildlife-conservation/inference/chip-n-scale-queue-arranger/base.py`
    - `ai4earth-wildlife-conservation/inference/chip-n-scale-queue-arranger/handler.py`

to the directory: `chip-n-scale-queue-arranger/lambda/download_and_predict`

Currently, the containerized model is located at [Development Seed's Dockerhub](https://hub.docker.com/r/devseeddeploy/aiaia_classifier/tags?page=1&ordering=last_updated)
<img width="1403" alt="Screen Shot 2021-01-19 at 11 11 10 AM" src="https://user-images.githubusercontent.com/14057932/105062169-4f544500-5a48-11eb-8083-0bc05317e463.png">

The average speed of the inference is **12,000 image chip/minute**. Each image chip is 400 x 400 pixel at 2-3 cm spatial resolution.

## Detector inferences

To run detector inference Copy and replace  the following files:
    - `ai4earth-wildlife-conservation/inference/chip-n-scale-queue-arranger/base.py`
    - `ai4earth-wildlife-conservation/inference/chip-n-scale-queue-arranger/handler.py`

to the directory: `chip-n-scale-queue-arranger/lambda/download_and_predict`

Currently, the containerized models are located at [Development Seed's Dockerhub](https://hub.docker.com/r/devseeddeploy/aiaia_fastrcnn/tags?page=1&ordering=last_updated)
<img width="1263" alt="Screen Shot 2021-01-19 at 11 11 46 AM" src="https://user-images.githubusercontent.com/14057932/105062388-8e829600-5a48-11eb-9fb1-12fdbdc4e62c.png">

The average speed of the inference is **600 image chip/minute**. Each image chip is 400 x 400 pixel at 2-3 cm spatial resolution.



### Setup Chip n Scale config env
Set the `config.yml` file and `.env` file
Add the config.yml file on `chip-n-scale-queue-arranger/config/`

config.yml
```
    default:
      stage: dev
      stackName: aiaia
      stackNoDash: aiaia
      capabilities:
         - CAPABILITY_NAMED_IAM
      buckets:
        internal: aiaia-inference # existing s3 bucket to store deployment artifacts
      lambdas:
        DownloadAndPredict:
          handler: download_and_predict.handler.handler
          timeout: 120
          memory: 1024
          source: lambda/package.zip
          runtime: python3.7
          queueTrigger: true
          concurrent: 2
          envs:
            BUCKET: wbg-geography01
            WB_AWS_ACCESS_KEY_ID: '{{AIAIA_AWS_ACCESS_KEY_ID}}'
            WB_AWS_SECRET_ACCESS_KEY: '{{AIAIA_AWS_SECRET_ACCESS_KEY}}'
      rds:
        username: '{{RDS_USERNAME}}'
        password: '{{RDS_PASSWORD}}'
        storage: 20
        instanceType: 'db.t2.medium'
      vpc: vpc-dfe524ba # existing VPC containing the two subnets below
      subnets:
        - subnet-XXX
        - subnet-XXX
      ecs:
        availabilityZone: us-east-1b
        maxInstances: 1
        desiredInstances: 1
        keyPairName: xxxx
        instanceType: p3.2xlarge  # replace with a GPU instance for faster predictions (and higher costs)
        image: devseeddeploy/aiaia_classifier:v1.0-gpu # docker image containing your inference model built with TF Serving
        memory: 40000 # replace with the memory required by your TF Serving docker image
        # edit the memory
      sqs:
        visibilityTimeout: 300 # visibility timeout to the maximum time that it takes your application to process and delete a message from the queue.
        maxReceiveCount: 5
      predictionPath: '/v1/models/xception_classifier' # path to your model on the TF Serving docker image; don't include :predict
```

- Notes:
    - Create your own bucket: `aiaia-inference`
    - Some times we need to increase the values , it depend the size of the image, for this case we are going to use a `400x400` image from s3  and a proper value is `lambdas/timeout: 120`  and `lambdas/memory: 1024`.

    - At the beginning we use  `concurrent: 2`, but later on once all is running ok, we could increase to `3,4,5`, it meas how many lambda functions will be triggered at the same time.
    - The value `visibilityTimeout: 300` should be equal or greather than  `lambdas/timeout: 120`
    - Don’t forget to set the image for GPU or CPU; `image: developmentseed/building_properties:v1-gpu` and also the path for the API:  `predictionPath: '/v1/models/xception_classifier'`
    - Use the instance type : `p2.xlarge` or it can be `p3.2xlarge`
    - Create your own Pairkey : `keyPairName: $your_key_pair`

  *Note*: if the lambda is not running out the memory, tune it to accept maximum image


Add the `.env` file on `chip-n-scale-queue-arranger/config/`

### Setup deployment .env`

```bash
    RDS_USERNAME=aiaiadb
    RDS_PASSWORD=xxx
    AIAIA_AWS_ACCESS_KEY_ID=xx
    AIAIA_AWS_SECRET_ACCESS_KEY=xxx
    BUCKET=aiaia-inference
```

Running the the infrastructure

```bash
cd chip-n-scale-queue-arranger/
cd lambda
make build


export AWS_PROFILE=devseed
cd chip-n-scale-queue-arranger/
nvm use v10.13.0
yarn install
yarn deploy
```

Note

In case there is an error on `yarn deploy`  update the `package.json`  file like so:


    "deploy": "AWS_SDK_LOAD_CONFIG=true kes cf deploy --kes-folder config --kes-class config/kes.js"

### Send the SQS to the stack

```
export AWS_PROFILE=hp
cd housing-passports/chip-n-scale-deployment/
```

update the  `get_s3_keys.py` with the right folders to get the path fo the images and then run the file

```bash

python get_s3_keys.py --profile_name=xxx \
                          --s3_bucket=xxx \
                          --s3_dir_prefix=cormonxxxx \
                          --out_txt=tiles.txt
```

 It going to create a file called `tiles.txt` with all the keys which we need to run the inference.

 For sending the messages execute:

```
python dap_send-images.py --imgs_txt_files tiles.txt
```


### Debugging on the AWS console


- Check the SQS :https://console.aws.amazon.com/sqs/home?region=us-east-1
    The stack creates two queues
        - AIAIADeadLetterQueue
        - AIAIATileQueue
    In case something goes wrong with the response of the lambda the messages will pass to the `AIAIADeadLetterQueue` .

- Check  cloudwatch: https://console.aws.amazon.com/cloudwatch/home?region=us-east-1 , to watch the lambda logs.


- Testing the API, run a small script: https://gist.github.com/Rub21/82851ddbacfbb49513ef59e7897148d4

Note:
Sometimes we need to rebuild the lambda function `cd lambda/ && make build`.


### Visualize the DB

Exact the model inference result from RDS PostgreSQL database.
