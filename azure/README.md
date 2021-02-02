## Azure Deployment Setup

1. Create the Blob Storage Container
```
bash create_blob_store.sh
```

2. Then move model and training data files from GCP to Azure. Or, from GCP to a local machine to using [AZCopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10).


3. Setup kubeflow on your local machine. This tools simplifies deploying containers with kubernetes instead of just using kubernetes.
```
bash setup_kubeflow.sh
```

4. Setup cluster. This might take 10 minutes or so.

```
bash setup_cluster.sh
```

###### Missing piece
open Kubeflow on the public IP and access to Kubeflow through the IP
######

5. After cluster is set up, we can submit the training job.
   
6. Once training is finished, we can delete the cluster and all associated resources.

```
bash delete_cluster.sh
```