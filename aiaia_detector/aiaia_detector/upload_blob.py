from azure.storage.blob import (
    BlobServiceClient,
)  # ideally we wouldn't need this and could use AzureML to sync outputs at the end of training
import os


def upload_file(client, source, dest):

    print(f"Uploading {source} to {dest}")
    with open(source, "rb") as data:
        client.upload_blob(name=dest, data=data)


def upload_dir(client, source, dest):

    prefix = "" if dest == "" else dest + "/"
    prefix += os.path.basename(source) + "/"
    for root, dirs, files in os.walk(source):
        for name in files:
            dir_part = os.path.relpath(root, source)
            dir_part = "" if dir_part == "." else dir_part + "/"
            file_path = os.path.join(root, name)
            blob_path = prefix + dir_part + name
            upload_file(client, file_path, blob_path)


def get_client(connect_str, container_name):
    try:

        service_client = BlobServiceClient.from_connection_string(connect_str)
        client = service_client.get_container_client(container_name)
        return client
    except Exception as ex:
        print("Exception:")
        print(ex)
