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
    name="aiaia-workspace-classifier",
    subscription_id=os.getenv("AZURE_SUB_ID"),
    resource_group="aiaia-workspace-classifier",
)


env = Environment("AIAIA Training")
env.docker.enabled = True
# Set the container registry information.
env.docker.base_image_registry.address = "aiaiatrain.azurecr.io"
env.docker.base_image_registry.username = "aiaiatrain"

env.docker.base_image = "v1.2-xception-binary-classifier:latest"
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
        vm_size="Standard_NC6s_v3", max_nodes=1,
    )
    # STANDARD_NC6 is cheaper but is a K80 and takes longer than a STANDARD_NC6s_v3 V100

    # Create the cluster.
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True)

# Use get_status() to get a detailed status for the current AmlCompute.
print(compute_target.get_status().serialize())

# Get a dataset by name
root_data_ds = Dataset.get_by_name(
    workspace=ws, name="root_data_for_azureml_detector"
)
dataset_input = root_data_ds.as_download(path_on_compute="/tmp/")
# outputs_ds = Dataset.get_by_name(workspace=ws, name="azureml_outputs")
# dataset_mount = outputs_ds.as_mount(path_on_compute="/mnt/")

src = ScriptRunConfig(
    source_directory="aiaia_detector",
    script="model_main.py",
    arguments=[
        "--root_data_path",
        dataset_input,
        "--model_dir",
        "logs",
        "--pipeline_config_path",
        "model_configs_tf1/configs/rcnn_resnet101_serengeti_wildlife.config",
        "--num_train_steps",
        "200",
        "--sample_1_of_n_eval_examples",
        "1",
        "--input_type",
        "image_tensor",
        "--output_directory",
        "outputs",
        "write_inference_graph",
        "True",
        "--connection_string",
        os.getenv("AZURE_CON_STRING"),
        "external_blob_container",
        os.getenv("BLOB_CONTAINER"),
        "external_blob_container_folder",
        "azureml_outputs_detector",
    ],
    compute_target=compute_target,
    environment=env,
)
# using "outputs" for --output_directory may cause latency issues
# using "logs" for --model_dir makes tensorboard profiling with azureml possible, but may cause latency issues
# both of these folders are mounts to blob storage that may continuously write files during the training process.
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
