import streamlit as st
import json
import os
import datetime
from loguru import logger
import pandas as pd
from utils import show_category_statistics

def load_coco_file(uploaded_file):
    """Load a COCO format JSON file from an uploaded file."""
    try:
        content = json.loads(uploaded_file.getvalue().decode('utf-8'))
        return content
    except Exception as e:
        st.error(f"Error loading COCO file: {e}")
        return None

def save_coco_file(coco_data, base_path="exported"):
    """Save the modified COCO data to a file."""
    today = datetime.datetime.now()
    now = today.strftime("%Y-%m-%d_%H:%M:%S")
    export_path = os.path.join(base_path, now)
    os.makedirs(export_path, exist_ok=True)
    
    annotations_dir = os.path.join(export_path, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    
    file_path = os.path.join(annotations_dir, "instances_default.json")
    
    with open(file_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    logger.info(f"Saved modified COCO file to {file_path}")
    return export_path, now



def main():
    st.write("# COCO Category Editor")
    st.write("Upload a COCO format annotation file to view and edit categories.")
    
    annotation_file = st.file_uploader("Upload COCO Annotation File", type=["json"])
    
    if "coco_data" not in st.session_state:
        st.session_state.coco_data = None
    
    if "task_path" not in st.session_state:
        st.session_state.task_path = None
        
    if "now" not in st.session_state:
        st.session_state.now = None
    
    if annotation_file:
        coco_data = load_coco_file(annotation_file)
        if coco_data:
            st.session_state.coco_data = coco_data
            st.success(f"Successfully loaded COCO annotation file: {annotation_file.name}")
    
    if st.session_state.coco_data:
        # Display statistics
        show_category_statistics(st.session_state.coco_data)
        
        st.divider()
        
        # Category editing section
        st.write("## Edit Categories")
        
        categories = st.session_state.coco_data.get('categories', [])
        
        # Create a DataFrame for the categories
        cat_df = pd.DataFrame(categories)
        
        # Create a form for editing categories
        with st.form("category_editor"):
            edited_df = st.data_editor(
                cat_df,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "id": st.column_config.NumberColumn(
                        "ID",
                        help="Category ID",
                        required=True,
                    ),
                    "name": st.column_config.TextColumn(
                        "Name",
                        help="Category name",
                        required=True,
                    ),
                    "supercategory": st.column_config.TextColumn(
                        "Supercategory",
                        help="Parent category name",
                    ),
                },
                hide_index=True,
            )
            
            submit_button = st.form_submit_button("Update Categories")
            
            if submit_button:
                # Check for duplicate IDs
                if edited_df['id'].duplicated().any():
                    st.error("Error: Duplicate category IDs found. Each category must have a unique ID.")
                else:
                    # Update the categories in the COCO data
                    st.session_state.coco_data['categories'] = edited_df.to_dict('records')
                    st.success("Categories updated successfully!")
                    
                    # Show updated statistics
                    show_category_statistics(st.session_state.coco_data)
        
        # Option to remove categories and their annotations
        st.write("## Remove Categories")
        st.write("Select categories to remove (this will also remove all associated annotations):")
        
        categories = st.session_state.coco_data.get('categories', [])
        category_options = {f"{cat['id']}: {cat['name']}": cat['id'] for cat in categories}
        
        selected_cats_to_remove = st.multiselect(
            "Categories to remove",
            options=list(category_options.keys()),
        )
        
        if st.button("Remove Selected Categories"):
            if selected_cats_to_remove:
                # Get the category IDs to remove
                cat_ids_to_remove = [category_options[cat_name] for cat_name in selected_cats_to_remove]
                
                # Filter out the selected categories
                st.session_state.coco_data['categories'] = [
                    cat for cat in st.session_state.coco_data['categories'] 
                    if cat['id'] not in cat_ids_to_remove
                ]
                
                # Filter out annotations for the removed categories
                st.session_state.coco_data['annotations'] = [
                    ann for ann in st.session_state.coco_data['annotations']
                    if ann['category_id'] not in cat_ids_to_remove
                ]
                
                st.success(f"Removed {len(selected_cats_to_remove)} categories and their annotations.")
                
                # Show updated statistics
                show_category_statistics(st.session_state.coco_data)
        
        # Save button
        st.divider()
        if st.button("Save Modified COCO File"):
            export_path, now = save_coco_file(st.session_state.coco_data)
            st.session_state.task_path = export_path
            st.session_state.now = now
            st.success(f"Modified COCO file saved to {export_path}")
            
            # Provide a download link
            with open(os.path.join(export_path, "annotations", "instances_default.json"), "r") as f:
                st.download_button(
                    label="Download Modified COCO File",
                    data=f,
                    file_name="modified_coco.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
