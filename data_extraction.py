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

