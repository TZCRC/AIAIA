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
# upload files 
CONTAINER=training
az storage blob upload \
    --account-name $ \
    --container-name $CONTAINER \
    --name training \
    --file train.records \
    --auth-mode login
