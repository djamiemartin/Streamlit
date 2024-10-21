import os

from google.cloud import storage

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "micro-root-379316-a3e3cb0be532.json"

storage_client = storage.Client()

dir(storage_client)


def download_file_from_bucket(blob_name, file_path, bucket_name):
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        with open(file_path, "wb") as f:
            storage_client.download_blob_to_file(blob, f)
        return True
    except Exception as e:
        print(e)
        return False


# create a function to download a folder from google cloud bucket


def download_folder(bucket_name, prefix, local_folder):
    client = storage.Client()

    bucket = client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix)

    for blob in blobs:
        relative_path = blob.name[len(prefix) :]

        # If relative_path is empty or represents a directory, skip it
        if not relative_path or relative_path.endswith("/"):
            continue

        # Create local destination path
        local_path = os.path.join(local_folder, relative_path)

        # Create directories if necessary
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download the file
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}")


bucket_name = "predict-maintenanceproject"
prefix = "dashboard/"  # Folder in the bucket
local_folder = "data/dashboard/"  # Folder locally
download_folder(bucket_name, prefix, local_folder)
