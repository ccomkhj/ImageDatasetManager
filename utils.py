import os
import datetime
from loguru import logger
import yaml
import boto3
import streamlit as st


def load_dataset_from_s3(s3_uri: str, local_download_path="existing_task"):
    credentials = load_aws_credentials()
    s3 = boto3.client(
        "s3",
        aws_access_key_id=credentials["aws_access_key_id"],
        aws_secret_access_key=credentials["aws_secret_access_key"],
    )

    if s3_uri.startswith("s3://"):
        s3_uri = s3_uri[5:]
    bucket_name, prefix = s3_uri.split("/", 1)

    os.makedirs(local_download_path, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            local_file_path = os.path.join(
                local_download_path, os.path.relpath(key, prefix)
            )
            local_dir = os.path.dirname(local_file_path)
            os.makedirs(local_dir, exist_ok=True)

            s3.download_file(bucket_name, key, local_file_path)
            logger.info(f"Downloaded {key} to {local_file_path}")

    return local_download_path


def load_dataset_from_s3_keep_parents(
    s3_uri: str, local_download_path="existing_task", file_extensions=None
):
    credentials = load_aws_credentials()
    s3 = boto3.client(
        "s3",
        aws_access_key_id=credentials["aws_access_key_id"],
        aws_secret_access_key=credentials["aws_secret_access_key"],
    )

    if s3_uri.startswith("s3://"):
        s3_uri = s3_uri[5:]

    bucket_name, prefix = s3_uri.split("/", 1)
    identifier = os.path.basename(s3_uri.rstrip("/"))
    local_download_path = os.path.join(local_download_path, identifier)
    os.makedirs(local_download_path, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    file_extensions = (
        file_extensions if file_extensions else [".jpg", ".jpeg", ".png", ".json"]
    )  # Default image extensions

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            logger.debug(f"Processing key: {key}")

            # Check if the file has the desired extension
            if not any(key.lower().endswith(ext) for ext in file_extensions):
                logger.info(f"Skipping non-target file type for {key}.")
                continue

            rel_path = os.path.relpath(key, prefix)
            local_file_path = os.path.join(local_download_path, rel_path)
            local_dir = os.path.dirname(local_file_path)

            try:
                os.makedirs(local_dir, exist_ok=True)
                logger.debug(f"Ensured directory exists: {local_dir}")
            except Exception as e:
                logger.error(f"Failed to create directory {local_dir}: {e}")
                continue

            if os.path.exists(local_file_path):
                logger.warning(
                    f"File already exists; skipping download: {local_file_path}"
                )
                continue

            try:
                s3.download_file(bucket_name, key, local_file_path)
                logger.info(f"Downloaded {key} to {local_file_path}")
            except Exception as e:
                logger.error(f"Failed to download {key}: {e}")

    return local_download_path


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
    annotation_path = os.path.join(annotations_path, "instances_default.json")
    with open(annotation_path, "wb") as f:
        f.write(annotation.getbuffer())

    return base_path, now


def load_aws_credentials(credentials_path="credentials/aws.yaml"):
    with open(credentials_path, "r") as stream:
        credentials = yaml.safe_load(stream)
    return credentials


def upload_to_s3(
    local_path,
    s3_uri,
    now: str,
    comment: str = "",
    credentials_path="credentials/aws.yaml",
):
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

        progress_text = f"Loading dataset ({root}) from S3 in progress."
        my_bar = st.progress(0, text=progress_text)
        total_files = len(files)
        uploaded_files = 0

        for file in files:
            local_file_path = os.path.join(root, file)
            s3_file_path = os.path.relpath(local_file_path, local_path)
            s3_key = os.path.join(prefix, f"{now}{comment}", s3_file_path)
            s3.upload_file(local_file_path, bucket_name, s3_key)
            logger.info(f"Uploaded {local_file_path} to s3://{bucket_name}/{s3_key}")
            my_bar.progress((uploaded_files) / total_files, text=progress_text)
            uploaded_files += 1

        my_bar.empty()
