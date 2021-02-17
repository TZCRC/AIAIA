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
        vm_size="STANDARD_NC6", max_nodes=1, idle_seconds_before_scaledown=1,
    )
    # STANDARD_NC6 is cheaper but is a K80 and takes longer than a STANDARD_NC6s_v3 V100

    # Create the cluster.
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    compute_target.wait_for_completion(show_output=True)

# Use get_status() to get a detailed status for the current AmlCompute.
print(compute_target.get_status().serialize())

# Get a dataset by name
root_data_ds = Dataset.get_by_name(
    workspace=ws, name="root_data_for_azureml_classifier"
)
dataset_input = root_data_ds.as_download(path_on_compute="/tmp/")

model_output_cfg = OutputFileDatasetConfig(
    name="model_output",
    destination=(
        ws.datastores[os.getenv("BLOB_CONTAINER")],
        "azureml_outputs_classifier/model_output",
    ),
).as_upload(overwrite=True)

model_results_cfg = OutputFileDatasetConfig(
    name="model_results",
    destination=(
        ws.datastores[os.getenv("BLOB_CONTAINER")],
        "azureml_outputs_classifier/model_results",
    ),
).as_upload(overwrite=True)


train_export_step = PythonScriptStep(
    name="Run training and export model",
    source_directory="aiaia_classifier",
    script_name="model.py",
    arguments=[
        "--n_classes=2",
        "--class_names=not_object,object",
        "--countries=aiaia",
        "--tf_dense_size=153",
        "--tf_dense_dropout_rate=0.34",
        "--tf_learning_rate=0.00027",
        "--tf_optimizer=adam",
        "--tf_train_data_dir=training_data_aiaia_p400/classification_training_tfrecords/",
        "--tf_val_data_dir=training_data_aiaia_p400/classification_training_tfrecords/",
        "--model_outputs_dir",
        model_output_cfg,
        "--local_dataset_dir",
        dataset_input,
        "--tf_steps_per_checkpoint=60",  # was 100
        "--tf_steps_per_summary=60",  # was 500
        "--tf_train_steps=60",  # was 6000
        "--tf_batch_size=8",
        "--results_dir",
        model_results_cfg,
        "--model_upload_id=v1",
        "--model_id=abc",  # put the unique tag for the experiment run here.
    ],
    compute_target=compute_target,
    runconfig=runconfig,
)
# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-save-write-experiment-files
# Create an experiment

steps = [train_export_step]

pl = Pipeline(workspace=ws, steps=steps)

pl.validate()

exp = Experiment(ws, "test_train_classifier_xception")

run = exp.submit(pl, regenerate_outputs=False)
run.wait_for_completion(show_output=True)
# tb = Tensorboard([run])

# # If successful, start() returns a string with the URI of the instance.
# tb.start()

# # Stops after experiment reaches "Completed" or "Failed"
# tb.stop()
