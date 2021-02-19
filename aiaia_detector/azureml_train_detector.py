from azureml.core import Workspace, Dataset
from azureml.core import Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Experiment
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.tensorboard import Tensorboard
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.core import Pipeline
import os

ws = Workspace.get(
    name="aiaia-workspace-detector",
    subscription_id=os.getenv("AZURE_SUB_ID"),
    resource_group="aiaia-workspace-detector",
)

env = Environment("AIAIA Training")
env.docker.enabled = True
# Set the container registry information.
env.docker.base_image_registry.address = "aiaiatrain.azurecr.io"
env.docker.base_image_registry.username = "aiaiatrain"

env.docker.base_image = "aiaia:v1-tf1.15-frozen-graph-gpu:latest"
env.python.user_managed_dependencies = True

# Run az acr credential show --name aiaiatrain to get credentials for the ACR.
env.docker.base_image_registry.password = os.getenv("AZURE_REGISTRY_PASSWORD")
runconfig = RunConfiguration()
runconfig.environment = env
# Choose a name for your cluster.
cluster_name = "gpu-cluster"

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print("Found existing compute target.")
except ComputeTargetException:
    print("Creating a new compute target...")
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="Standard_NC6s_v3",
        max_nodes=1,
        idle_seconds_before_scaledown=1,
    )
    # STANDARD_NC6 is cheaper but is a K80 and takes longer than a STANDARD_NC6s_v3 V100

    # Create the cluster.
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True)

# Use get_status() to get a detailed status for the current AmlCompute.
print(compute_target.get_status().serialize())

#### we wrap paths in azureml objects to handle transferring data to and from the cluster with
#### Dataset.as_download() and OutputFileDatasetConfig.as_upload()
# https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.data.output_dataset_config.outputfiledatasetconfig?view=azure-ml-py

# Get a dataset by name
root_data_ds = Dataset.get_by_name(
    workspace=ws, name="root_data_for_azureml_detector"
)
dataset_input = root_data_ds.as_download(path_on_compute="/tmp/")

model_logs_output_cfg = OutputFileDatasetConfig(
    name="model_logs",
    destination=(
        ws.datastores[os.getenv("BLOB_CONTAINER")],
        "azureml_outputs_detector/model_logs",
    ),
).as_upload(overwrite=True)


model_output_cfg = OutputFileDatasetConfig(
    name="model_outputs",
    destination=(
        ws.datastores[os.getenv("BLOB_CONTAINER")],
        "azureml_outputs_detector/model_output",
    ),
).as_upload(overwrite=True)
####

train_export_step = PythonScriptStep(
    name="Run training and export model",
    source_directory="aiaia_detector",
    script_name="model_main.py",
    arguments=[
        "--root_data_path",
        dataset_input,
        "--model_dir",
        model_logs_output_cfg,
        "--pipeline_config_path",
        "model_configs_tf1/configs/rcnn_resnet101_serengeti_wildlife.config",
        "--num_train_steps",
        "10",  # used to be 50000
        "--sample_1_of_n_eval_examples",
        "1",
        "--input_type",
        "image_tensor",
        "--output_directory",
        model_output_cfg,
        "write_inference_graph",
        "True",
    ],
    compute_target=compute_target,
    runconfig=runconfig,
)
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-save-write-experiment-files
# Create an experiment

steps = [train_export_step]

pl = Pipeline(workspace=ws, steps=steps)

pl.validate()

exp = Experiment(ws, "test_train_rcnn_resnet101_serengeti_wildlife")

run = exp.submit(pl)
run.wait_for_completion(show_output=True)
tb = Tensorboard([run])

# If successful, start() returns a string with the URI of the instance.
tb.start()

# Stops after experiment reaches "Completed" or "Failed"
tb.stop()
