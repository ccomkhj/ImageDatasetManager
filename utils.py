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


def show_category_statistics(coco_data):
    """
    Display statistics about categories and their annotations in a COCO format dataset.
    
    Args:
        coco_data (dict): The COCO format data containing categories, annotations, and images.
    """
    if not coco_data:
        return
    
    categories = coco_data.get('categories', [])
    annotations = coco_data.get('annotations', [])
    images = coco_data.get('images', [])
    
    # Basic statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Categories", len(categories))
    with col2:
        st.metric("Total Images", len(images))
    with col3:
        st.metric("Total Annotations", len(annotations))
    
    # Count annotations per category
    from collections import Counter
    category_counts = Counter([ann['category_id'] for ann in annotations])
    
    # Create a mapping from category_id to category name
    category_map = {cat['id']: cat['name'] for cat in categories}
    
    # Create a DataFrame for better visualization
    import pandas as pd
    stats_data = []
    for cat in categories:
        cat_id = cat['id']
        stats_data.append({
            'Category ID': cat_id,
            'Category Name': cat['name'],
            'Annotation Count': category_counts.get(cat_id, 0)
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Count unique images per category
    images_per_category = {}
    for ann in annotations:
        cat_id = ann['category_id']
        img_id = ann['image_id']
        
        if cat_id not in images_per_category:
            images_per_category[cat_id] = set()
        
        images_per_category[cat_id].add(img_id)
    
    # Create DataFrame for images per category
    img_stats_data = []
    for cat_id, img_set in images_per_category.items():
        img_stats_data.append({
            'Category ID': cat_id,
            'Category Name': category_map.get(cat_id, f"Unknown ({cat_id})"),
            'Image Count': len(img_set)
        })
    
    img_stats_df = pd.DataFrame(img_stats_data)
    
    # Display in two columns
    st.write("### Category Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        # Display the statistics table
        st.dataframe(stats_df, use_container_width=True)
        
        # Create a bar chart of annotation counts
        if not stats_df.empty:
            import matplotlib.pyplot as plt
            import numpy as np
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(stats_df['Category Name'], stats_df['Annotation Count'])
            
            # Add the values on top of the bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.xlabel('Category')
            plt.ylabel('Number of Annotations')
            plt.title('Annotations per Category')
            
            st.pyplot(fig)
    
    with col2:
        # Images per category
        st.write("### Images per Category")
        
        if not img_stats_df.empty:
            st.dataframe(img_stats_df, use_container_width=True)
            
            # Create a bar chart of image counts
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(img_stats_df['Category Name'], img_stats_df['Image Count'])
            
            # Add the values on top of the bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.xlabel('Category')
            plt.ylabel('Number of Images')
            plt.title('Images per Category')
            
            st.pyplot(fig)

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
    import random

    categories = dataset.categories()[dm.AnnotationType.label]
    num_categories = len(categories)

    # Generate visually distinct colors with higher saturation and value
    def get_distinct_colors(n):
        colors = []
        # Use golden ratio to create well-distributed hues
        golden_ratio_conjugate = 0.618033988749895
        h = random.random()  # Starting hue
        
        for i in range(n):
            h = (h + golden_ratio_conjugate) % 1
            # Higher saturation (0.9) and value (1.0) for more vibrant colors
            rgb = colorsys.hsv_to_rgb(h, 0.9, 1.0)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors

    category_colors = get_distinct_colors(max(num_categories, 1))  # Ensure at least one color

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
            elif hasattr(ann, "label_id") and ann.label_id is not None:
                # Some annotation formats use label_id instead of label
                label_id = ann.label_id
                if label_id < len(category_colors):
                    color = category_colors[label_id]
                    # Try to get label name from categories
                    for cat in categories:
                        if hasattr(cat, 'id') and cat.id == label_id:
                            label_name = cat.name
                            break
            # Dictionary to group annotations by their group ID
            # This helps us visualize related annotations (like bbox and keypoints) together
            annotations_by_group = {}
            
            if ann.type == dm.AnnotationType.bbox:
                x1, y1, x2, y2 = ann.points
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw label text with background for better visibility
                if label_name:
                    # Get text size for background
                    from PIL import ImageFont
                    try:
                        font = ImageFont.load_default()
                        text_width, text_height = draw.textsize(label_name, font=font)
                    except:
                        # Fallback if textsize is not available
                        text_width, text_height = len(label_name) * 8, 15
                    
                    # Draw text background
                    draw.rectangle(
                        [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                        fill=color
                    )
                    # Draw text in white for contrast
                    draw.text((x1 + 2, y1 - text_height - 2), label_name, fill=(255, 255, 255))
                
                # Check if there are keypoints with the same group_id as this bbox
                if hasattr(ann, 'group') and ann.group is not None:
                    group_id = ann.group
                    # Look for keypoint annotations with the same group_id
                    for keypoint_ann in item.annotations:
                        if (keypoint_ann.type == dm.AnnotationType.points and 
                            hasattr(keypoint_ann, 'group') and 
                            keypoint_ann.group == group_id):
                            # Draw keypoints with the same color as the bounding box
                            point_size = 8  # Larger point size
                            outline_size = 2  # White outline size
                            for px, py in zip(keypoint_ann.points[::2], keypoint_ann.points[1::2]):
                                # Draw white outline first
                                draw.ellipse(
                                    [px - point_size - outline_size, py - point_size - outline_size,
                                     px + point_size + outline_size, py + point_size + outline_size],
                                    fill=(255, 255, 255),
                                    outline=(255, 255, 255)
                                )
                                # Then draw the colored point
                                draw.ellipse(
                                    [px - point_size, py - point_size, px + point_size, py + point_size],
                                    fill=color,
                                    outline=color
                                )
            elif ann.type == dm.AnnotationType.label:
                if label_name:
                    draw.text((10, 10), label_name, fill=color)
            elif ann.type == dm.AnnotationType.points:
                # If this keypoint annotation doesn't have a matching bbox (no group_id or no matching bbox),
                # we'll draw it independently
                has_matching_bbox = False
                if hasattr(ann, 'group') and ann.group is not None:
                    group_id = ann.group
                    for bbox_ann in item.annotations:
                        if (bbox_ann.type == dm.AnnotationType.bbox and 
                            hasattr(bbox_ann, 'group') and 
                            bbox_ann.group == group_id):
                            has_matching_bbox = True
                            break
                
                # Only draw keypoints here if they don't have a matching bbox
                if not has_matching_bbox:
                    point_size = 8  # Larger point size
                    outline_size = 2  # White outline size
                    for px, py in zip(ann.points[::2], ann.points[1::2]):
                        # Draw white outline first
                        draw.ellipse(
                            [px - point_size - outline_size, py - point_size - outline_size,
                             px + point_size + outline_size, py + point_size + outline_size],
                            fill=(255, 255, 255),
                            outline=(255, 255, 255)
                        )
                        # Then draw the colored point
                        draw.ellipse(
                            [px - point_size, py - point_size, px + point_size, py + point_size],
                            fill=color,
                            outline=color
                        )
            # Add more annotation types as needed
        st.image(img, caption=f"{item.id}")
        count += 1
        if count >= max_items:
            break