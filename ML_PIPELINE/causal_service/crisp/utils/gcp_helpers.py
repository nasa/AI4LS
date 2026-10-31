import json
from io import BytesIO, StringIO

import pandas as pd
from google.cloud import storage
from google.oauth2 import service_account


def get_byte_fileobj(project: str,
                     bucket: str,
                     path: str,
                     service_account_credentials_path: str = None) -> BytesIO:
    """
    Retrieve data from a given blob on Google Storage and pass it as a file object.
    :param path: path within the bucket
    :param project: name of the project
    :param bucket_name: name of the bucket
    :param service_account_credentials_path: path to credentials.
           TIP: can be stored as env variable, e.g. os.getenv('GOOGLE_APPLICATION_CREDENTIALS_DSPLATFORM')
    :return: file object (BytesIO)
    """
    blob = _get_blob(bucket, path, project, service_account_credentials_path)
    byte_stream = BytesIO()
    blob.download_to_file(byte_stream)
    byte_stream.seek(0)
    return byte_stream


def get_bytestring(project: str,
                   bucket: str,
                   path: str,
                   service_account_credentials_path: str = None) -> bytes:
    """
    Retrieve data from a given blob on Google Storage and pass it as a byte-string.
    :param path: path within the bucket
    :param project: name of the project
    :param bucket_name: name of the bucket
    :param service_account_credentials_path: path to credentials.
           TIP: can be stored as env variable, e.g. os.getenv('GOOGLE_APPLICATION_CREDENTIALS_DSPLATFORM')
    :return: byte-string (needs to be decoded)
    """
    blob = _get_blob(bucket, path, project, service_account_credentials_path)
    s = blob.download_as_string()
    return s


def _get_blob(bucket_name, path, project, service_account_credentials_path):
    credentials = service_account.Credentials.from_service_account_file(
        service_account_credentials_path) if service_account_credentials_path else None
    storage_client = storage.Client(project=project, credentials=None)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(path)
    return blob


def write_csv_to_blob(data, blob):
    stream = StringIO()
    data.to_csv(stream, sep=";", index=False)
    blob.upload_from_string(stream.getvalue())


def save_dataframe_to_bucket(data, savepath, project, bucket):
    #     credentials = service_account.Credentials.from_service_account_file(
    #         service_account_credentials_path) if service_account_credentials_path else None
    storage_client = storage.Client(project=project, credentials=None)
    bucket = storage_client.get_bucket(bucket)
    blob = bucket.blob(savepath)
    write_csv_to_blob(data, blob)
    return 1


def write_json_to_blob(data, blob):
    blob.upload_from_string(
        data=json.dumps(data),
        content_type='application/json'
    )
    return 1


def save_json_to_bucket(data, savepath, project, bucket):
    storage_client = storage.Client(project=project, credentials=None)
    bucket = storage_client.get_bucket(bucket)
    blob = bucket.blob(savepath)
    write_json_to_blob(data, blob)
    return 1


def get_json_from_bucket(filepath1, project, bucket):
    fileobj = get_byte_fileobj(project, bucket, filepath1)
    # Load the JSON to a Python list & dump it back out as formatted JSON
    data = json.load(fileobj)
    return data


def get_dataframe_from_bucket(filepath1, project, bucket):
    fileobj = get_byte_fileobj(project, bucket, filepath1)
    df = pd.read_csv(fileobj, sep=";")
    return df


def list_files_in_bucket_folder(gcp_folder, project, bucket):
    storage_client = storage.Client(project=project, credentials=None)
    files = []
    for blob in storage_client.list_blobs(bucket, prefix=gcp_folder):
        files.append(blob.name)
    files.remove(gcp_folder)
    return files
