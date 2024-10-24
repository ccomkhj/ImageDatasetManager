import streamlit as st
import datumaro as dm
from datumaro.components.dataset import Dataset
from datumaro.components.hl_ops import HLOps
import datumaro.plugins.splitter as splitter
from utils import (
    save_uploaded_files,
    upload_to_s3,
    load_dataset_from_s3,
    load_dataset_from_s3_keep_parents,
)
import os


def merge_and_split_datasets(
    new_task_path, existing_datasets, now, output_base_path="merged", split=True
):
    # Load the new dataset
    new_dataset = Dataset.import_from(new_task_path, "coco_instances")

    # Check if all categories are the same across all datasets
    new_label_names = [
        label_cat.name
        for label_cat in new_dataset.categories()[dm.AnnotationType.label]
    ]

    # Merge datasets
    merged_dataset = new_dataset

    for existing_dataset_path in existing_datasets:

        # Load the existing dataset
        existing_dataset = Dataset.import_from(existing_dataset_path, "coco_instances")
        existing_label_names = [
            label_cat.name
            for label_cat in existing_dataset.categories()[dm.AnnotationType.label]
        ]
        if new_label_names != existing_label_names:
            raise ValueError("Categories of the datasets do not match!")

        # Perform merging using union policy
        merged_dataset = merged_dataset.update(existing_dataset)

    # Aggregate subsets to one default
    aggregated = HLOps.aggregate(
        merged_dataset, from_subsets=merged_dataset.subsets(), to_subset="default"
    )

    if split:
        # Split the aggregated dataset
        splits = [("train", 0.8), ("val", 0.2)]
        task = splitter.SplitTask.detection.name
        aggregated = aggregated.transform("split", task=task, splits=splits)

    os.makedirs(output_base_path, exist_ok=True)
    export_path = os.path.join(output_base_path, now)
    aggregated.export(export_path, "coco_instances", reindex=True, save_media=True)

    return export_path


def main():
    st.write("# Merge Annotations")
    dataset_source = st.radio("Where are the existing datasets?", ("S3", "Local"))

    existing_s3_uris = ""
    prev_images = None
    prev_annotation = None

    if dataset_source == "S3":
        existing_s3_uris = st.text_area(
            "Enter S3 URIs for the existing tasks (one per line) (e.g., s3://existing-task-folder/)"
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
    split_option = st.checkbox("Split after merging?", value=True)  # Add this line

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
        if (dataset_source == "S3" and existing_s3_uris) or (
            dataset_source == "Local" and prev_images and prev_annotation
        ):
            new_task_path, now = save_uploaded_files(images, annotation)
            existing_task_paths = []

            if dataset_source == "S3":
                s3_uris = existing_s3_uris.strip().split("\n")
                for s3_uri in s3_uris:
                    if s3_uri:
                        existing_task_path = load_dataset_from_s3_keep_parents(
                            s3_uri.strip(),
                            file_extensions=[".jpg", ".jpeg", ".png", ".json"],
                        )
                        existing_task_paths.append(existing_task_path)
            elif dataset_source == "Local":
                # Save existing dataset
                existing_task_path, _ = save_uploaded_files(
                    prev_images, prev_annotation
                )
                existing_task_paths.append(existing_task_path)

            # Merge datasets and split based on user choice
            merged_task_path = merge_and_split_datasets(
                new_task_path, existing_task_paths, now, split=split_option
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
