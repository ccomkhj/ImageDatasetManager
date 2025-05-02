import os
import datetime
from loguru import logger
import yaml
import boto3
import streamlit as st
from PIL import Image, ImageDraw
import numpy as np

from datumaro.util.image import IMAGE_BACKEND, ImageBackend

IMAGE_BACKEND.set(ImageBackend.PIL)


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


def save_uploaded_files(images, annotation, job_type):
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

    match job_type:
        case "keypoints":
            annotation_path = os.path.join(
                annotations_path, "person_keypoints_default.json"
            )
        case "segmentation":
            annotation_path = os.path.join(annotations_path, "stuff_default.json")
        case _:
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


def visualize_dataset_with_annotations(dataset_path, dataset_type="coco", max_items=10):
    """
    Visualize images with their annotations from a Datumaro dataset using Streamlit.
    Args:
        dataset_path (str): Path to the exported dataset (root folder).
        dataset_type (str): Datumaro dataset type (e.g., 'coco', 'voc').
        max_items (int): Maximum number of images to visualize.
    """
    import datumaro as dm
    from datumaro.components.dataset import Dataset

    try:
        dataset = Dataset.import_from(dataset_path, dataset_type)
    except Exception as e:
        st.error(f"Failed to load dataset for visualization: {e}")
        return

    st.write(f"## Visualizing up to {max_items} images with annotations")
    count = 0
    from datumaro.util.image import IMAGE_BACKEND, ImageBackend

    IMAGE_BACKEND.set(ImageBackend.PIL)
    # Generate a color palette for categories
    import colorsys

    categories = dataset.categories()[dm.AnnotationType.label]
    num_categories = len(categories)

    # Generate visually distinct colors
    def get_n_colors(n):
        return [
            tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / n, 0.8, 0.95))
            for i in range(n)
        ]

    category_colors = get_n_colors(num_categories)

    for item in dataset:
        # Load image as PIL
        if hasattr(item.media, "data") and item.media.data is not None:
            img = item.media.data
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif not isinstance(img, Image.Image):
                st.warning(f"Cannot load image for item {item.id}")
                continue
        else:
            st.warning(f"Item {item.id} has unsupported or missing media.")
            continue

        draw = ImageDraw.Draw(img)
        for ann in item.annotations:
            color = (255, 0, 0)  # default red
            label_name = None
            if (
                hasattr(ann, "label")
                and ann.label is not None
                and ann.label < num_categories
            ):
                label_name = categories[ann.label].name
                color = category_colors[ann.label]
            if ann.type == dm.AnnotationType.bbox:
                x1, y1, x2, y2 = ann.points
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                # Draw label text at top-left
                if label_name:
                    draw.text((x1 + 2, y1 + 2), label_name, fill=color)
            elif ann.type == dm.AnnotationType.label:
                if label_name:
                    draw.text((10, 10), label_name, fill=color)
            elif ann.type == dm.AnnotationType.points:
                for px, py in zip(ann.points[::2], ann.points[1::2]):
                    draw.ellipse(
                        [px - 2, py - 2, px + 2, py + 2], fill=color, outline=color
                    )
            # Add more annotation types as needed
        st.image(img, caption=f"{item.id}")
        count += 1
        if count >= max_items:
            break
