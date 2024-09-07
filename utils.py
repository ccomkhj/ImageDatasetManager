import os
import datetime
from loguru import logger
import yaml
import boto3


def save_uploaded_files(images, annotation):
    today = datetime.datetime.now()
    now = today.strftime("%Y-%m-%d_%H:%M:%S")
    base_path = f"uploaded/{now}"
    logger.info(f"Saving uploaded files at {base_path}")
    images_path = os.path.join(base_path, "images/")
    annotations_path = os.path.join(base_path, "annotations/")

    os.makedirs(images_path, exist_ok=True)
    os.makedirs(annotations_path, exist_ok=True)

    # Save image files
    for image in images:
        with open(os.path.join(images_path, image.name), "wb") as f:
            f.write(image.getbuffer())

    # Save annotation file
    annotation_path = os.path.join(annotations_path, annotation.name)
    with open(annotation_path, "wb") as f:
        f.write(annotation.getbuffer())

    return base_path, now


def load_aws_credentials(credentials_path="credentials/aws.yaml"):
    with open(credentials_path, "r") as stream:
        credentials = yaml.safe_load(stream)
    return credentials


def upload_to_s3(local_path, s3_uri, credentials_path="credentials/aws.yaml"):
    credentials = load_aws_credentials(credentials_path)

    # Split the S3 URI into bucket name and prefix
    if s3_uri.startswith("s3://"):
        s3_uri = s3_uri[5:]
    bucket_name, prefix = s3_uri.split("/", 1)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=credentials["aws_access_key_id"],
        aws_secret_access_key=credentials["aws_secret_access_key"],
    )

    for root, _, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_file_path = os.path.relpath(local_file_path, local_path)
            s3_key = os.path.join(prefix, s3_file_path)
            s3.upload_file(local_file_path, bucket_name, s3_key)
            logger.info(f"Uploaded {local_file_path} to s3://{bucket_name}/{s3_key}")
