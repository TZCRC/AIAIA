from azureml.core import Workspace, Dataset, Datastore
import os

ws = Workspace.create(
    name="aiaia",
    subscription_id=os.getenv("AZURE_SUB_ID"),
    resource_group="aiaia-workspace",
    create_resource_group=False,
    location="eastus",
)

blob_datastore_name = os.getenv(
    "BLOB_CONTAINER"
)  # Name of the datastore to workspace
container_name = os.getenv("BLOB_CONTAINER")  # Name of Azure blob container
account_name = os.getenv("BLOB_ACCOUNTNAME")  # Storage account name
account_key = os.getenv("BLOB_ACCOUNT_KEY")  # Storage account access key

blob_datastore = Datastore.register_azure_blob_container(
    workspace=ws,
    datastore_name=blob_datastore_name,
    container_name=container_name,
    account_name=account_name,
    account_key=account_key,
)

ws.set_default_datastore(os.getenv("BLOB_CONTAINER"))

# create a FileDataset pointing to files in 'animals' folder and its subfolders recursively
datastore_paths = [(blob_datastore, "root_data_for_azureml")]
root_data_ds = Dataset.File.from_files(path=datastore_paths)
root_data_ds = root_data_ds.register(
    workspace=ws,
    name="root_data_for_azureml",
    description="tfrecords and model files for training classifier and object detectors. Also includes tfrecords for testing and master model, which are unused in training the classifier and three object detection models.",
    create_new_version=True,
)

datastore_paths = [(blob_datastore, "azureml_outputs")]
outputs_ds = Dataset.File.from_files(path=datastore_paths)
outputs_ds = outputs_ds.register(
    workspace=ws,
    name="azureml_outputs",
    description="folder to save checkpoint and frozen graph outputs.",
    create_new_version=True,
)


ws.write_config(path="./.azureml", file_name="ws_config.json")
