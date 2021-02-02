# use correct resource group and location
RESOURCE_GROUP_NAME=$1
LOCATION=$2

# use correct storage acount
STORAGE_ACC=$3

# use container created during create_blob_store_step
CONTAINER=training

# create a specifically defined cluster
AKS_NAME=kubeflow-aks #cannot exceed 63 characters and can only contain letters, numbers, or dashes (-).
AGENT_SIZE=Standard_NC6 #this machine type can add GPU driver.
AGENT_COUNT=2

#this will take about 10mins
az aks create -g ${RESOURCE_GROUP_NAME} -n ${AKS_NAME} -s ${AGENT_SIZE} -c ${AGENT_COUNT} -l ${LOCATION} --generate-ssh-keys

# update acr for aks cluster, may take awhile
ACR_NAME=$3
az acr create -n $ACR_NAME -g $RESOURCE_GROUP_NAME --sku basic
# update ACR for existing AKS cluster, this may take awhile
az aks update -n $AKS_NAME -g $RESOURCE_GROUP_NAME --attach-acr $ACR_NAME

#Adding GPU driver to AKS
#get the AKS credentials 
az aks get-credentials --resource-group $RESOURCE_GROUP_NAME --name $AKS_NAME 
kubectl create namespace gpu-resources
# create a nvidia-device-plugin-ds.yaml shows as following
kubectl apply -f nvidia-device-plugin-ds.yaml

# confirm GPUs schedulable
kubectl get nodes

# Kubeflow deploy

# The following command is optional, to make kfctl binary easier to use.
export PATH=$PATH:${HOME}/.kfctl

# Set KF_NAME to the name of your Kubeflow deployment. This also becomes the
# name of the directory containing your configuration.
# For example, your deployment name can be 'my-kubeflow' or 'kf-test'.
export KF_NAME=kf-ml-aiaia

# Set the path to the base directory where you want to store one or more 
# Kubeflow deployments. For example, /opt/.
# Then set the Kubeflow application directory for this deployment.
export BASE_DIR=$PWD
export KF_DIR=${BASE_DIR}/${KF_NAME}

# Set the configuration file to use, such as the file specified below:
export CONFIG_URI="https://raw.githubusercontent.com/kubeflow/manifests/v1.0-branch/kfdef/kfctl_k8s_istio.v1.0.2.yaml"

# Generate and deploy Kubeflow:
mkdir -p ${KF_DIR}
cd ${KF_DIR}
# empty files under KF_DIR before deploy kubeflow
kfctl apply -V -f ${CONFIG_URI}

# check if kubectl set up correctly
kubectl get all -n kubeflow

# Access Control for Azure Kubeflow Deployment
# Switching the ingressgateway service to be a LoadBalancer
kubectl patch service -n istio-system istio-ingressgateway -p '{"spec": {"type": "LoadBalancer"}}'
echo "Find your IP and add the following under the type property, which is located in the selector section."
echo "loadBalancerSourceRanges:\n-  <your-ip>/32"
echo "# Then save the file `ESCAPE` then `:wq`, if it worked you should see:"
echo "service/istio-ingressgateway edited"
wait 5
kubectl edit svc -n istio-system istio-ingressgateway
echo "To get the IP of your deployments load balancer, run: "
echo "kubectl get svc -n istio-system istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0]}'"
echo "Your IP is:"
kubectl get svc -n istio-system istio-ingressgateway -o jsonpath='{.status.loadBalancer.ingress[0]}'
