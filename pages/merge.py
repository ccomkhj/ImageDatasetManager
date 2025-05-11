import streamlit as st
import pandas as pd
import datumaro as dm
from datumaro.components.dataset import Dataset
from datumaro.components.hl_ops import HLOps
import datumaro.plugins.splitter as splitter
from utils import (
    save_uploaded_files,
    upload_to_s3,
    load_dataset_from_s3,
    load_dataset_from_s3_keep_parents,
    visualize_dataset_with_annotations,
    show_category_statistics,
)
import os
import json


def merge_and_split_datasets(
    new_task_path, existing_datasets, now, output_base_path="merged", split=True, job_type=None
):
    # Note: it is aggregating instead of merging. So it assumes homogeneous dataset
    
    # Map job_type to the correct type_name
    if job_type is None:
        type_name = "coco"
    elif job_type == "instances":
        type_name = "coco_instances"
    elif job_type == "keypoints":
        type_name = "coco_person_keypoints"
    elif job_type == "segmentation":
        type_name = "coco_stuff"
    else:
        # Default to coco if job_type is not recognized
        type_name = "coco"
    
    # Load the new dataset
    new_dataset = Dataset.import_from(new_task_path, type_name)

    # Check if all categories are the same across all datasets
    new_label_names = [
        label_cat.name
        for label_cat in new_dataset.categories()[dm.AnnotationType.label]
    ]

    # Merge datasets
    merged_dataset = new_dataset

    for existing_dataset_path in existing_datasets:

        # Load the existing dataset
        existing_dataset = Dataset.import_from(existing_dataset_path, type_name)
        existing_label_names = [
            label_cat.name
            for label_cat in existing_dataset.categories()[dm.AnnotationType.label]
        ]
        if set(new_label_names) != set(existing_label_names):
            raise ValueError("Categories of the datasets do not match!")

        # Perform merging using union policy
        merged_dataset = merged_dataset.update(existing_dataset)

    # Aggregate subsets to one default
    aggregated = HLOps.aggregate(
        merged_dataset, from_subsets=merged_dataset.subsets(), to_subset="default"
    )
    
    # Create a temporary export for statistics before splitting
    temp_export_path = os.path.join(output_base_path, f"{now}_temp")
    os.makedirs(temp_export_path, exist_ok=True)
    
    # Export the aggregated dataset for statistics
    aggregated.export(temp_export_path, type_name, reindex=True, save_media=False)
    
    # Get the annotation file path for statistics
    if job_type == "keypoints":
        annotation_filename = "person_keypoints_default.json"
    elif job_type == "segmentation":
        annotation_filename = "stuff_default.json"
    else:  # instances or default
        annotation_filename = "instances_default.json"
        
    stats_annotations_path = os.path.join(temp_export_path, "annotations", annotation_filename)
    
    # Now prepare the actual export with splitting if needed
    os.makedirs(output_base_path, exist_ok=True)
    export_path = os.path.join(output_base_path, now)
    
    if split:
        # Split the aggregated dataset
        splits = [("train", 0.8), ("val", 0.2)]
        aggregated = aggregated.transform("random_split", splits=splits)

    # Export the final dataset
    aggregated.export(export_path, type_name, reindex=True, save_media=True)
    
    return export_path, stats_annotations_path


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
    split_option = st.checkbox("Split after merging?", value=True)

    st.divider()

    images = st.file_uploader(
        "Upload Image Files for New Task",
        accept_multiple_files=True,
        type=["jpg", "png"],
    )
    annotation = st.file_uploader(
        "Upload Annotation File for New Task", type=["json", "xml"]
    )

    # Add the dropdown menu for selecting job type
    job_type = st.selectbox(
        "Select Annotation Type", ["instances", "keypoints", "segmentation"]
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
            new_task_path, now = save_uploaded_files(images, annotation, job_type)
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
                    prev_images, prev_annotation, job_type
                )
                existing_task_paths.append(existing_task_path)

            # Merge datasets and split based on user choice
            merged_task_path, annotations_path = merge_and_split_datasets(
                new_task_path, existing_task_paths, now, split=split_option, job_type=job_type
            )

            st.session_state.merged_task_path = merged_task_path
            st.success(
                f"Datasets merged and split. Merged data saved at {merged_task_path}"
            )
            st.session_state.now = now
            
            # Show category statistics after merging
            st.divider()
            st.write("## Category Statistics After Merging")
            try:
                with open(annotations_path, 'r', encoding='utf-8') as f:
                    coco_data = json.load(f)
                show_category_statistics(coco_data)
            except Exception as e:
                st.error(f"Error loading COCO file for statistics: {e}")

    # Visualization option after merging
    if st.session_state.get("merged_task_path"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Visualize Merged Annotations"):
                visualize_dataset_with_annotations(
                    st.session_state["merged_task_path"], "coco"
                )
            st.caption("You can visually check the merged annotations above.")
        
        with col2:
            if st.button("Show Category Statistics"):
                st.divider()
                st.write("## Category Statistics")
                try:
                    annotations_path = os.path.join(st.session_state.merged_task_path + "_temp", "annotations", "instances_default.json")
                    with open(annotations_path, 'r', encoding='utf-8') as f:
                        coco_data = json.load(f)
                    show_category_statistics(coco_data)
                except Exception as e:
                    st.error(f"Error loading COCO file for statistics: {e}")

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
                
                # Show category statistics after upload
                st.divider()
                st.write("## Category Statistics of Uploaded Data")
                try:
                    annotations_path = os.path.join(st.session_state.merged_task_path, "annotations", "instances_default.json")
                    with open(annotations_path, 'r', encoding='utf-8') as f:
                        coco_data = json.load(f)
                    show_category_statistics(coco_data)
                except Exception as e:
                    st.error(f"Error loading COCO file for statistics: {e}")


main()