echo "Deleting Cluster and all resources"
RESOURCE_GROUP_NAME=$1
CLUSTER_NAME=kubeflow-aks
kubectl get all -n kubeflow
# takes a few minutes
az aks delete --name kubeflow-aks --resource-group $RESOURCE_GROUP_NAME