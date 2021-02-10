from azureml.core import Workspace, Dataset
from azureml.core import Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Experiment
from azureml.core import ScriptRunConfig
from azureml.tensorboard import Tensorboard
import os
import time

ws = Workspace.get(
    name="aiaia",
    subscription_id=os.getenv("AZURE_SUB_ID"),
    resource_group="aiaia-workspace",
)


env = Environment("AIAIA Training")
env.docker.enabled = True
# Set the container registry information.
env.docker.base_image_registry.address = "aiaiatrain.azurecr.io"
env.docker.base_image_registry.username = "aiaiatrain"

env.docker.base_image = "aiaia-tf1.15-frozen_graph:latest"
env.python.user_managed_dependencies = True

# Run az acr credential show --name aiaiatrain to get credentials for the ACR.
env.docker.base_image_registry.password = os.getenv("AZURE_REGISTRY_PASSWORD")


# Choose a name for your cluster.
cluster_name = "gpu-cluster"

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print("Found existing compute target.")
except ComputeTargetException:
    print("Creating a new compute target...")
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_NC6",
        max_nodes=1,
    )

    # Create the cluster.
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True)

# Use get_status() to get a detailed status for the current AmlCompute.
print(compute_target.get_status().serialize())

# Get a dataset by name
tfrecord_ds = Dataset.get_by_name(workspace=ws, name="tfrecord_train_ds")
dataset_input = tfrecord_ds.as_download(path_on_compute="./")

src = ScriptRunConfig(
    source_directory="aiaia_detector",
    script="model_main.py",
    arguments=[
        "--model_dir",
        "/tmp/training_data_aiaia_p400/model_outputs_tf1/rcnn_resnet101_serengeti_wildlife",
        "--pipeline_config_path",
        "/tmp/training_data_aiaia_p400/model_configs_tf1/configs/rcnn_resnet101_serengeti_wildlife.config",
        "--num_train_steps",
        "50000",
        "--sample_1_of_n_eval_examples",
        "1",
        "--input_type",
        "image_tensor",
        "--output_directory",
        "/tmp/training_data_aiaia_p400/export_outputs_tf1/rcnn_resnet101_serengeti_wildlife",
    ],
    compute_target=compute_target,
    environment=env,
)
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-save-write-experiment-files
# Create an experiment
exp = Experiment(ws, "test_train_rcnn_resnet101_serengeti_wildlife")

run = exp.submit(src)
run.wait_for_completion(show_output=True)
# tb = Tensorboard([run])

# # If successful, start() returns a string with the URI of the instance.
# tb.start()

# while run.get_status() is "Running":
#     time.sleep(10)

# # Stops after experiment reaches "Completed" or "Failed"
# tb.stop()