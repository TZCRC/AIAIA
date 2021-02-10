## Azure Deployment Setup

1. Create the Blob Storage Container and Azure Container Registry.  Set AZURE_STORAGE_KEY before running
    ```
    bash create_acr_fileshare.sh aiaiatrain eastus aiaiatrain
    ```
    - Make sure you have already logged into the azure cli with `az login`.
    - Note: The current raw training dataset is about 1 Tb as measured from AWS S3. Azure Blob Store can't be mounted easily and file IOPs are limted for both Blob and File stores. File Storage can be used for training models using datasets that are a few gigabytes in size, [but not much larger](https://github.com/Azure/kubeflow-labs/tree/master/10-going-further). For this reason we need to move all the data to a data disk that is mounted to the VM prior to training.
    - see `az account list-locations -o table` if you want to set up a container in an availability region closer to you, which may increase transfer speeds from your local machine.
    - If you need to build a new image for whatever region and reupload to acr:
    ```
    docker build -t aiaiatrain.azurecr.io/aiaia-tf1.15-frozen_graph -f Dockerfile-gpu .

    # wait while it builds locally, then

    docker push aiaiatrain.azurecr.io/aiaia-tf1.15-frozen_graph

    ```

2. Move the frozen graph models, model configs, and processed training data TFRecords from GCP to Azure. Or, from a local machine to Azure to using [AZCopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10).
   -  before using azcopy to copy from gcp > azure, make a SAS token in the azure portal byt going to Storage Accounts > gcsaiaiatrain > Shared Access signature in the side menu.
   -  also, set `GOOGLE_APPLICATION_CREDENTIALS` and `GOOGLE_CLOUD_PROJECT`
   -  then with azcopy version 10.9 or higher, `azcopy cp "https://storage.cloud.google.com/aiaia_od/csv_files" "https://gcsaiaiatrain.blob.core.windows.net/training/?<SAS Token from azure portal>" --recursive=true`
   -  For transferring from aws, set up [aws credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-profiles.html) if you are [moving from aws s3 to azure blob storage](https://azure.microsoft.com/en-us/blog/move-your-data-from-aws-s3-to-azure-storage-using-azcopy/). You can also use azcopy to move data from your local machine to Azure blob storage.
   -  you'll need to specify the bucket location on s3. get it with `aws s3api get-bucket-location --bucket aisurvey`
   -  then
   ```
   azcopy copy 'https://<region>.amazonaws.com/<bucket>/' 'https://aiaiatrain.blob.core.windows.net/aiaiatrain<SAS>' --recursive=true
   ```

3. Setup AzureML libraries and workspace config on your local machine
   1. Create a Workspace in the Azure portal.
   2. https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-environment#workspace
      1. create the configuration at `AIAIA/aiaia_detector/.azureml/config.json`
   3. The following three commands get past a weird bug in ruamel_yaml install. 
      1. `conda create -n azureml python=3.6.2 pip=20.1.1`
      2. `conda activate azureml`
      3. `pip install azureml-sdk` on your local machine to get the libraries needed to create a workspace and run experiments. Activate the environment when running python files using azureml.

4. 