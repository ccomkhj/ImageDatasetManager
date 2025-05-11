import streamlit as st
import os
import json
from enhanced_viz import enhanced_category_statistics, visualize_coco_annotations_sample
from utils import load_dataset_from_s3

def main():
    st.write("# Dataset Statistics Visualization")
    st.write("""
    This page provides advanced visualization and statistics for your COCO format datasets.
    Upload a COCO annotation file or select from an existing dataset to view enhanced visualizations.
    """)
    
    dataset_source = st.radio("Dataset Source", ("Local Upload", "Local Path", "S3"))
    
    coco_data = None
    image_dir = None
    
    if dataset_source == "Local Upload":
        annotation_file = st.file_uploader("Upload COCO Annotation File", type=["json"])
        images_folder = st.text_input("Path to Images Folder (optional)", "")
        
        if annotation_file:
            try:
                coco_data = json.loads(annotation_file.getvalue().decode('utf-8'))
                st.success(f"Successfully loaded COCO annotation file: {annotation_file.name}")
                
                if images_folder and os.path.exists(images_folder):
                    image_dir = images_folder
                    st.success(f"Using images from: {image_dir}")
                else:
                    st.warning("Images folder not specified or not found. Some visualizations may be limited.")
            except Exception as e:
                st.error(f"Error loading COCO file: {e}")
    
    elif dataset_source == "Local Path":
        dataset_path = st.text_input("Path to Dataset Folder", "")
        
        if dataset_path and os.path.exists(dataset_path):
            # Try to find annotation file
            annotations_dir = os.path.join(dataset_path, "annotations")
            if os.path.exists(annotations_dir):
                annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
                
                if annotation_files:
                    selected_file = st.selectbox("Select Annotation File", annotation_files)
                    
                    try:
                        with open(os.path.join(annotations_dir, selected_file), 'r') as f:
                            coco_data = json.load(f)
                        st.success(f"Successfully loaded COCO annotation file: {selected_file}")
                        
                        # Look for images directory
                        images_dir = os.path.join(dataset_path, "images")
                        if os.path.exists(images_dir):
                            image_dir = images_dir
                            st.success(f"Using images from: {image_dir}")
                        else:
                            st.warning("Images folder not found. Some visualizations may be limited.")
                    except Exception as e:
                        st.error(f"Error loading COCO file: {e}")
                else:
                    st.warning("No annotation files found in the specified path.")
            else:
                st.warning("No 'annotations' directory found in the specified path.")
        else:
            st.warning("Please enter a valid dataset path.")
    
    elif dataset_source == "S3":
        s3_uri = st.text_input("Enter S3 URI (e.g., s3://bucket-name/path/to/dataset/)", "")
        
        if s3_uri and st.button("Load Dataset from S3"):
            try:
                local_path = load_dataset_from_s3(s3_uri)
                st.success(f"Dataset downloaded from S3 to {local_path}")
                
                # Try to find annotation file
                annotations_dir = os.path.join(local_path, "annotations")
                if os.path.exists(annotations_dir):
                    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
                    
                    if annotation_files:
                        selected_file = st.selectbox("Select Annotation File", annotation_files)
                        
                        try:
                            with open(os.path.join(annotations_dir, selected_file), 'r') as f:
                                coco_data = json.load(f)
                            st.success(f"Successfully loaded COCO annotation file: {selected_file}")
                            
                            # Look for images directory
                            images_dir = os.path.join(local_path, "images")
                            if os.path.exists(images_dir):
                                image_dir = images_dir
                                st.success(f"Using images from: {image_dir}")
                            else:
                                st.warning("Images folder not found. Some visualizations may be limited.")
                        except Exception as e:
                            st.error(f"Error loading COCO file: {e}")
                    else:
                        st.warning("No annotation files found in the downloaded dataset.")
                else:
                    st.warning("No 'annotations' directory found in the downloaded dataset.")
            except Exception as e:
                st.error(f"Error loading dataset from S3: {e}")
    
    # Display enhanced visualizations if data is loaded
    if coco_data:
        st.divider()
        
        # Show enhanced category statistics
        enhanced_category_statistics(coco_data, image_dir=image_dir)
        
        # Option to view sample annotations
        if image_dir:
            st.divider()
            if st.button("Show Detailed Sample Annotations"):
                visualize_coco_annotations_sample(coco_data, image_dir=image_dir, max_samples=5)
        else:
            st.warning("Image directory not available. Cannot show sample annotations.")

if __name__ == "__main__":
    main()
