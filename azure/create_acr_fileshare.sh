# create a resource group
RESOURCE_GROUP_NAME=$1
LOCATION=$2
az group create -n ${RESOURCE_GROUP_NAME} -l ${LOCATION}

# create storage acount
STORAGE_ACC=$3
az storage account create \
    --name $STORAGE_ACC \
    --resource-group $RESOURCE_GROUP_NAME \
    --location $LOCATION \
    --sku Standard_ZRS \
    --encryption-services blob

# create a container
# at the portal-azure 
# set AZURE_STORAGE_KEY before running
FILESHARE=training
az storage share create \
    --account-name $STORAGE_ACC \
    --name $FILESHARE \

# create azure container registry
ACR_NAME=aiaiatrain
az acr create -n $ACR_NAME -g $RESOURCE_GROUP_NAME --sku basic
