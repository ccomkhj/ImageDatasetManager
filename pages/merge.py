import streamlit as st
import datumaro as dm
from datumaro.components.dataset import Dataset
from datumaro.components.hl_ops import HLOps
import datumaro.plugins.splitter as splitter
from utils import (
    save_uploaded_files,
    upload_to_s3,
    load_dataset_from_s3,
)
import os


def merge_and_split_datasets(
    new_task_path, existing_task_path, now, output_base_path="merged"
):
    # Load datasets
    new_dataset = Dataset.import_from(new_task_path, "coco")
    existing_dataset = Dataset.import_from(existing_task_path, "coco")

    # Check if categories are the same
    new_label_names = [
        label_cat.name
        for label_cat in new_dataset.categories()[dm.AnnotationType.label]
    ]
    existing_label_names = [
        label_cat.name
        for label_cat in existing_dataset.categories()[dm.AnnotationType.label]
    ]

    if new_label_names != existing_label_names:
        raise ValueError("Categories of the datasets do not match!")

    # Merge datasets
    merged_dataset = HLOps.merge(existing_dataset, new_dataset, merge_policy="union")

    # Aggregate subsets
    aggregated = HLOps.aggregate(
        merged_dataset, from_subsets=["default"], to_subset="default"
    )

    # Split the aggregated dataset
    splits = [("train", 0.8), ("val", 0.2)]
    task = splitter.SplitTask.detection.name
    resplitted = aggregated.transform("split", task=task, splits=splits)

    os.makedirs(output_base_path, exist_ok=True)
    export_path = os.path.join(output_base_path, now)
    resplitted.export(export_path, "coco", save_media=True)

    return export_path


def main():
    st.write("# Merge Annotations")

    dataset_source = st.radio("Where is the existing dataset?", ("S3", "Local"))

    existing_s3_uri = ""
    prev_images = None
    prev_annotation = None
    if dataset_source == "S3":
        existing_s3_uri = st.text_input(
            "Enter S3 URI for the existing task (e.g., s3://existing-task-folder/)"
        )
    elif dataset_source == "Local":
        st.info("Please upload the existing dataset files (images and annotations).")

        prev_images = st.file_uploader(
            "Upload Existing Image Files",
            accept_multiple_files=True,
            type=["jpg", "png"],
        )
        prev_annotation = st.file_uploader(
            "Upload Existing Annotation File", type=["json", "xml"]
        )
        st.divider()

    images = st.file_uploader(
        "Upload Image Files for New Task",
        accept_multiple_files=True,
        type=["jpg", "png"],
    )
    annotation = st.file_uploader(
        "Upload Annotation File for New Task", type=["json", "xml"]
    )

    if "merged_task_path" not in st.session_state:
        st.session_state.merged_task_path = None
    if "s3_uri" not in st.session_state:
        st.session_state.s3_uri = ""
    if "s3_comment" not in st.session_state:
        st.session_state.s3_comment = ""

    if st.button("Merge Datasets"):
        if (dataset_source == "S3" and existing_s3_uri) or (
            dataset_source == "Local" and prev_images and prev_annotation
        ):
            new_task_path, now = save_uploaded_files(images, annotation)

            # Load the existing dataset
            if dataset_source == "S3":
                existing_task_path = load_dataset_from_s3(existing_s3_uri)
            elif dataset_source == "Local":
                # Save existing dataset
                existing_task_path, _ = save_uploaded_files(
                    prev_images, prev_annotation
                )

            # Merge datasets and split again
            merged_task_path = merge_and_split_datasets(
                new_task_path, existing_task_path, now
            )
            st.session_state.merged_task_path = merged_task_path
            st.success(
                f"Datasets merged and split. Merged data saved at {merged_task_path}"
            )
            st.session_state.now = now

    if st.session_state.merged_task_path is not None:
        st.session_state.s3_uri = st.text_input(
            "Enter S3 URI to upload (e.g., s3://hexa-cv-dataset/test/)",
            st.session_state.s3_uri,
        )
        st.session_state.s3_comment = st.text_input(
            "Enter Comment for S3 dataset (e.g., cross_validation)",
            st.session_state.s3_comment,
        )

        if st.button("Upload Merged Dataset to S3"):
            if not st.session_state.merged_task_path:
                st.warning("First merge the datasets before uploading to S3.")
            elif not st.session_state.s3_uri:
                st.warning("Write S3 URI.")
            else:
                upload_to_s3(
                    st.session_state.merged_task_path,
                    st.session_state.s3_uri,
                    st.session_state.now,
                    st.session_state.s3_comment,
                )
                st.success(
                    f"Merged dataset uploaded to S3 at {st.session_state.s3_uri}"
                )


main()
