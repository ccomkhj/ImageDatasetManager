import streamlit as st
import pandas as pd
import datumaro as dm
from datumaro.components.dataset import Dataset
from datumaro.plugins.validators import (
    DetectionValidator,
)
import os
import tempfile
import boto3
from botocore.exceptions import NoCredentialsError
import pandas as pd
from utils import load_dataset_from_s3_keep_parents


# Helper functions based on your provided structure
def save_uploaded_files(images, annotation):
    """
    Save uploaded image files and annotation to a temporary directory.

    Returns:
    - base_path: Path to the directory where files are saved.
    - now: Current timestamp as a string.
    """
    now = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    base_path = tempfile.mkdtemp(prefix="dataset_" + now + "_")

    # Save images
    images_dir = os.path.join(base_path, "images")
    os.makedirs(images_dir, exist_ok=True)
    for image_file in images:
        image_path = os.path.join(images_dir, image_file.name)
        with open(image_path, "wb") as f:
            f.write(image_file.getbuffer())

    # Save annotation
    annotations_dir = os.path.join(base_path, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    annotation_path = os.path.join(annotations_dir, annotation.name)
    with open(annotation_path, "wb") as f:
        f.write(annotation.getbuffer())

    return base_path, now


def load_dataset_from_s3(s3_uri, file_extensions=None):
    """
    Download dataset files from S3 to a temporary directory.

    Parameters:
    - s3_uri: S3 URI of the dataset.
    - file_extensions: List of file extensions to filter files.

    Returns:
    - base_path: Path to the directory where files are saved.
    """
    if file_extensions is None:
        file_extensions = [".jpg", ".jpeg", ".png", ".json", ".xml", ".txt"]

    # Parse S3 URI
    if s3_uri.startswith("s3://"):
        s3_uri = s3_uri[5:]
    bucket_name, s3_path = s3_uri.split("/", 1)
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=s3_path)

    base_path = tempfile.mkdtemp(prefix="dataset_s3_")
    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                # Skip directories or unwanted file types
                if key.endswith("/"):
                    continue
                if not any(key.lower().endswith(ext) for ext in file_extensions):
                    continue
                local_path = os.path.join(base_path, os.path.relpath(key, s3_path))
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                s3.download_file(bucket_name, key, local_path)
    return base_path


def visualize_reports(reports):
    """
    Visualizes the given reports dictionary in Streamlit.

    Parameters:
    - reports (dict): The reports dictionary containing summary, statistics, and validation reports.
    """

    # 1. Display Summary Metrics
    st.header("Validation Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Errors", reports["summary"].get("errors", 0))
    col2.metric("Warnings", reports["summary"].get("warnings", 0))
    col3.metric("Infos", reports["summary"].get("infos", 0))

    # 2. Show Statistics
    st.header("Statistics")
    statistics = reports.get("statistics", {})
    for stat_name, stat_value in statistics.items():
        if stat_name in [
            "attribute_distribution",
            "point_distribution_in_label",
            "point_distribution_in_dataset_item",  # Exclude this as per your request
        ]:
            continue
        st.subheader(stat_name.replace("_", " ").title())
        if isinstance(stat_value, dict):
            if stat_value:  # Check if the dictionary is not empty
                df = pd.DataFrame(
                    list(stat_value.items()), columns=[stat_name.title(), "Count"]
                )
                st.table(df)
            else:
                st.write("No data available.")
        elif isinstance(stat_value, list):
            st.write(stat_value)
        else:
            st.write(stat_value)

    # 3. Render Validation Reports Efficiently
    st.header("Validation Reports")
    validation_reports = reports.get("validation_reports", [])

    if validation_reports:
        # Create a DataFrame to summarize validation issues
        report_rows = []
        for report in validation_reports:
            severity = report.get("severity", "Unknown")
            anomaly_type = report.get("anomaly_type", "")
            description = report.get("description", "")
            item_id = report.get("item_id", "")
            report_rows.append(
                {
                    "Severity": severity,
                    "Anomaly Type": anomaly_type,
                    "Description": description,
                    "Item ID": item_id,
                }
            )

        df_reports = pd.DataFrame(report_rows)

        # Map severities to colors or icons
        severity_colors = {
            "error": "🔴",
            "warning": "🟠",
            "info": "🟢",
            "unknown": "⚪️",
        }

        df_reports["Severity Icon"] = df_reports["Severity"].apply(
            lambda x: severity_colors.get(x.lower(), "⚪️")
        )

        # Rearrange columns
        df_reports = df_reports[
            [
                "Severity Icon",
                "Severity",
                "Anomaly Type",
                "Item ID",
                "Description",
            ]
        ]

        # Display the DataFrame with styling
        st.dataframe(df_reports)

    else:
        st.write("No validation issues found.")


def main():
    st.title("Dataset Validation Application")

    st.write("## Select Dataset Source")
    dataset_source = st.radio("Dataset Source", ("Local", "S3"))

    if dataset_source == "Local":
        st.write("### Upload your dataset")
        images = st.file_uploader(
            "Upload Image Files",
            accept_multiple_files=True,
            type=["jpg", "jpeg", "png"],
        )
        annotation = st.file_uploader(
            "Upload Annotation File", type=["json", "xml", "txt", "csv"]
        )
        if st.button("Validate Dataset"):
            if images and annotation:
                # Save uploaded files
                base_path, now = save_uploaded_files(images, annotation)
                # Load dataset
                try:
                    dataset = Dataset.import_from(
                        base_path, "coco_instances"
                    )  # Assuming COCO format
                    # Validate dataset
                    validator = DetectionValidator()
                    # Limit the types of validations to make it computationally cheaper
                    reports = validator.validate(dataset)
                    # Visualize report
                    visualize_reports(reports)
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")
            else:
                st.warning("Please upload both images and annotation file.")
    elif dataset_source == "S3":
        st.write("### Enter S3 Details")
        s3_uri = st.text_input("Enter S3 URI (e.g., s3://your-bucket/path/)")
        if st.button("Validate Dataset"):
            if s3_uri:
                try:
                    # Load dataset from S3
                    base_path = load_dataset_from_s3_keep_parents(s3_uri)
                    # Load dataset
                    dataset = Dataset.import_from(
                        base_path, "coco_instances"
                    )  # Assuming COCO format
                    # Validate dataset
                    validator = DetectionValidator()
                    # Limit the types of validations to make it computationally cheaper
                    reports = validator.validate(dataset)
                    # Visualize report
                    visualize_reports(reports)
                except NoCredentialsError:
                    st.error(
                        "AWS Credentials are not found. Please configure your AWS credentials."
                    )
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")
            else:
                st.warning("Please provide the S3 URI.")


if __name__ == "__main__":
    main()
