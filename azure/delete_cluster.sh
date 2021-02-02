echo "Deleting Cluster and all resources"
RESOURCE_GROUP_NAME=$1
kubectl get all -n kubeflow
az group delete -n $RESOURCE_GROUP_NAME