import streamlit as st
import datumaro as dm
from datumaro.components.dataset import Dataset
from datumaro.components.hl_ops import HLOps
import datumaro.plugins.splitter as splitter
from utils import save_uploaded_files, upload_to_s3, visualize_dataset_with_annotations, show_category_statistics
import os
import json


def create_new_task_split(
    input_base_path: str, now: str, export_base_path="exported", job_type=None
):

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
        # Default to coco_instances if job_type is not recognized
        type_name = "coco_instances"

    # Load new dataset
    dataset = Dataset.import_from(input_base_path, type_name)

    # Aggregate subsets
    aggregated = HLOps.aggregate(dataset, from_subsets=["default"], to_subset="default")
    
    # Create a temporary export for statistics before splitting
    temp_export_path = os.path.join(export_base_path, f"{now}_temp")
    os.makedirs(temp_export_path, exist_ok=True)
    
    # Export the aggregated dataset for statistics
    aggregated.export(temp_export_path, type_name, save_media=False)
    
    # Get the annotation file path for statistics
    if job_type == "keypoints":
        annotation_filename = "person_keypoints_default.json"
    elif job_type == "segmentation":
        annotation_filename = "stuff_default.json"
    else:  # instances or default
        annotation_filename = "instances_default.json"
        
    stats_annotations_path = os.path.join(temp_export_path, "annotations", annotation_filename)
    
    # Now split the dataset for the actual export
    splits = [("train", 0.8), ("val", 0.2)]
    resplitted = aggregated.transform("random_split", splits=splits)

    export_path = os.path.join(export_base_path, now)

    # Export the split datasets
    resplitted.export(export_path, type_name, save_media=True) # reindex = True??
    
    return export_path, stats_annotations_path


def main():
    st.write("# Register Annotation")
    images = st.file_uploader(
        "Upload Image Files", accept_multiple_files=True, type=["jpg", "png"]
    )
    annotation = st.file_uploader("Upload Annotation File", type=["json", "xml"])

    # Add the dropdown menu for selecting job type
    job_type = st.selectbox(
        "Select Annotation Type", ["instances", "keypoints", "segmentation"]
    )

    # Add the dropdown menu for selecting dataset type
    dataset_types = ["coco_instances", "coco", "voc", "cityscapes", "ade20k"]
    dataset_type = st.selectbox("Select Dataset Type", dataset_types, index=0)

    if "task_path" not in st.session_state:
        st.session_state.task_path = None
    if "s3_uri" not in st.session_state:
        st.session_state.s3_uri = ""
    if "s3_comment" not in st.session_state:
        st.session_state.s3_comment = ""

    if st.button("Register Annotation"):
        if images and annotation:
            base_path, now = save_uploaded_files(images, annotation, job_type)
            task_path, annotations_path = create_new_task_split(base_path, now, job_type=job_type)
            st.session_state.task_path = task_path
            st.success(f"New task created at {task_path}.")
            st.session_state.now = now
            
            # Show category statistics after creating new task
            st.divider()
            st.write("## Category Statistics")
            try:
                if os.path.exists(annotations_path):
                    with open(annotations_path, 'r', encoding='utf-8') as f:
                        coco_data = json.load(f)
                    # Get the images directory path
                    images_dir = os.path.join(task_path, "images")
                    show_category_statistics(coco_data, image_dir=images_dir)
                else:
                    st.warning(f"Annotation file not found at {annotations_path}")
            except Exception as e:
                st.error(f"Error loading COCO file for statistics: {e}")
        else:
            st.warning("No file upload. Check both images and annotation.")

    # Visualization option after registration
    if st.session_state.get("task_path"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Visualize Annotations"):
                visualize_dataset_with_annotations(
                    st.session_state["task_path"], dataset_type
                )
            st.caption("You can visually check the registered annotations above.")
        
        with col2:
            if st.button("Show Category Statistics"):
                st.divider()
                st.write("## Category Statistics")
                try:
                    # Try to find the annotation file based on job_type
                    annotation_files = [
                        "instances_default.json",
                        "person_keypoints_default.json",
                        "stuff_default.json"
                    ]
                    
                    annotations_dir = os.path.join(st.session_state.task_path + "_temp", "annotations")
                    found_file = False
                    
                    for filename in annotation_files:
                        file_path = os.path.join(annotations_dir, filename)
                        if os.path.exists(file_path):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                coco_data = json.load(f)
                            show_category_statistics(coco_data)
                            found_file = True
                            break
                    
                    if not found_file:
                        st.warning(f"No annotation files found in {annotations_dir}")
                except Exception as e:
                    st.error(f"Error loading COCO file for statistics: {e}")

    if st.session_state.task_path:
        st.session_state.s3_uri = st.text_input(
            "Enter S3 URI to upload (e.g., s3://hexa-cv-dataset/test/)",
            value=st.session_state.s3_uri,
            on_change=lambda: setattr(
                st.session_state, "s3_uri", st.session_state.s3_uri
            ),
        )
        st.session_state.s3_comment = st.text_input(
            "Enter Comment for S3 dataset (e.g., cross_validation)",
            value=st.session_state.s3_comment,
            on_change=lambda: setattr(
                st.session_state, "s3_comment", st.session_state.s3_comment
            ),
        )
        if st.button("Upload to S3"):
            if len(st.session_state.s3_uri) == 0:
                st.warning("Write S3 URI.")
            else:
                upload_to_s3(
                    st.session_state.task_path,
                    st.session_state.s3_uri,
                    st.session_state.now,
                    st.session_state.s3_comment,
                )
                st.success(f"Data uploaded to S3 at {st.session_state.s3_uri}")
                
                # Show category statistics after upload
                st.divider()
                st.write("## Category Statistics of Uploaded Data")
                try:
                    # Try to find the annotation file based on job_type
                    annotation_files = [
                        "instances_default.json",
                        "person_keypoints_default.json",
                        "stuff_default.json"
                    ]
                    
                    annotations_dir = os.path.join(st.session_state.task_path, "annotations")
                    found_file = False
                    
                    for filename in annotation_files:
                        file_path = os.path.join(annotations_dir, filename)
                        if os.path.exists(file_path):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                coco_data = json.load(f)
                            show_category_statistics(coco_data)
                            found_file = True
                            break
                    
                    if not found_file:
                        st.warning(f"No annotation files found in {annotations_dir}")
                except Exception as e:
                    st.error(f"Error loading COCO file for statistics: {e}")
    elif st.button("Upload to S3"):
        st.warning("First create a task before uploading to S3.")


main()
