import os

from google.cloud import storage

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'micro-root-379316-a3e3cb0be532.json'

storage_client = storage.Client()

dir(storage_client)

def download_file_from_bucket(blob_name, file_path, bucket_name):
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        with open(file_path, 'wb') as f:
            storage_client.download_blob_to_file(blob, f)
        return True
    except Exception as e:
        print(e)
        return False

bucket_name = 'predict-maintenanceproject'
download_file_from_bucket('Predictive/Predictive-csv1/c1/c1/c_1_001.csv', os.path.join(os.getcwd(), 'c_1_001.csv'), bucket_name)

# create a function to download a folder from google cloud bucket
from google.cloud import storage
import os

def download_folder(bucket_name, prefix, local_folder):
    # Create a storage client
    client = storage.Client()

    # Get the bucket
    bucket = client.get_bucket(bucket_name)

    # List all objects in the folder (prefix)
    blobs = bucket.list_blobs(prefix=prefix)

    # Download each file in the folder
    for blob in blobs:
        # Get the relative path within the folder
        relative_path = blob.name[len(prefix):]
        # Create local destination path
        local_path = os.path.join(local_folder, relative_path)

        # Create directories if necessary
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download the file
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob.name} to {local_path}")

# Usage
#bucket_name = 'predict-maintenanceproject'
#prefix =   'Predictive/Predictive-csv1/c1/c1/'    #'your-folder-prefix/'  # Folder in the bucket
#local_folder =  'streamlit'                            #'/local/destination/folder'
#download_folder(bucket_name, prefix, local_folder)
#download_folder(bucket_name, Predictive/Predictive-csv1/c1/c1/, streamlit)


