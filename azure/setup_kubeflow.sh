# use correct resource group and location
RESOURCE_GROUP_NAME=$1
# Kubeflow setup > install > deploy
#Create user credentials. You only need to run this command once.
az aks get-credentials -n ${AKS_NAME} -g ${RESOURCE_GROUP_NAME}
# download kubeflow v1.0.2 https://github.com/kubeflow/kfctl/releases/tag/v1.0.2
platform=0-ga476281_darwin
tar -xvf kfctl_v1.0.2-${platform}.tar.gz -C ~/.kfctl  
