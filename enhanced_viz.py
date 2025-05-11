import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image as PILImage
import numpy as np
import io
import colorsys
import random
import os
from collections import Counter
import pandas as pd

def visualize_coco_annotations_sample(coco_data, image_dir=None, max_samples=3):
    """
    Visualize sample COCO annotations with enhanced visualization.
    
    Args:
        coco_data (dict): The COCO format data containing categories, annotations, and images.
        image_dir (str, optional): Directory containing the images. If None, will try to find images.
        max_samples (int): Maximum number of sample images to show per category.
    """
    if not coco_data:
        st.warning("No COCO data provided for visualization.")
        return
    
    categories = coco_data.get('categories', [])
    annotations = coco_data.get('annotations', [])
    images = coco_data.get('images', [])
    
    if not categories or not annotations or not images:
        st.warning("COCO data is missing essential components (categories, annotations, or images).")
        return
    
    # Create category ID to name mapping
    category_map = {cat['id']: cat['name'] for cat in categories}
    
    # Generate visually distinct colors for categories
    def get_distinct_colors(n):
        colors = []
        golden_ratio_conjugate = 0.618033988749895
        h = random.random()  # Starting hue
        
        for i in range(n):
            h = (h + golden_ratio_conjugate) % 1
            # Higher saturation (0.9) and value (1.0) for more vibrant colors
            rgb = colorsys.hsv_to_rgb(h, 0.9, 1.0)
            colors.append(tuple(int(c * 255) for c in rgb))
        return colors
    
    category_colors = {cat['id']: get_distinct_colors(1)[0] for cat in categories}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Group images by category
    images_by_category = {}
    for img_id, anns in annotations_by_image.items():
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in images_by_category:
                images_by_category[cat_id] = set()
            images_by_category[cat_id].add(img_id)
    
    # Find image directory if not provided
    if image_dir is None:
        # Try common parent directories
        possible_dirs = ['images', 'train', 'val', 'test']
        for dir_name in possible_dirs:
            if os.path.exists(dir_name):
                image_dir = dir_name
                break
    
    # Create a mapping from image_id to file_name
    image_map = {img['id']: img['file_name'] for img in images}
    
    # Function to draw annotations on an image
    def draw_annotations(img_array, annotations):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img_array)
        ax.axis('off')
        
        # Draw each annotation
        for ann in annotations:
            cat_id = ann['category_id']
            cat_name = category_map.get(cat_id, f"Unknown ({cat_id})")
            color = category_colors.get(cat_id, (255, 0, 0))  # Default to red if not found
            color_rgb = [c/255 for c in color]  # Convert to 0-1 range for matplotlib
            
            # Handle different annotation types
            if 'bbox' in ann:
                # COCO bbox format is [x, y, width, height]
                x, y, w, h = ann['bbox']
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color_rgb, facecolor='none')
                ax.add_patch(rect)
                
                # Add label with background for better visibility
                text_box = patches.Rectangle((x, y-20), len(cat_name)*10, 20, 
                                            facecolor=color_rgb, alpha=0.8)
                ax.add_patch(text_box)
                ax.text(x+5, y-10, cat_name, color='white', fontweight='bold')
            
            if 'segmentation' in ann and ann['segmentation']:
                # Handle segmentation masks
                for segment in ann['segmentation']:
                    # Convert flat list to x,y pairs
                    poly = np.array(segment).reshape(-1, 2)
                    polygon = patches.Polygon(poly, closed=True, 
                                             fill=True, color=color_rgb, alpha=0.4,
                                             linewidth=2, edgecolor=color_rgb)
                    ax.add_patch(polygon)
            
            if 'keypoints' in ann and len(ann['keypoints']) > 0:
                # Draw keypoints
                keypoints = np.array(ann['keypoints']).reshape(-1, 3)
                for kp in keypoints:
                    x, y, v = kp
                    if v > 0:  # Only draw visible keypoints
                        ax.plot(x, y, 'o', markersize=8, markerfacecolor=color_rgb, 
                                markeredgecolor='white', markeredgewidth=2)
        
        # Convert the plot to an image for Streamlit
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig)
        return buf
    
    # Display samples for each category
    st.write("## Annotation Samples by Category")
    
    # Use tabs for categories
    if len(categories) > 0:
        tabs = st.tabs([cat['name'] for cat in categories])
        
        for i, cat in enumerate(categories):
            with tabs[i]:
                cat_id = cat['id']
                cat_name = cat['name']
                
                # Get images for this category
                img_ids = list(images_by_category.get(cat_id, set()))
                
                if not img_ids:
                    st.write(f"No images found for category: {cat_name}")
                    continue
                
                # Limit to max_samples
                sample_img_ids = random.sample(img_ids, min(max_samples, len(img_ids)))
                
                st.write(f"### Sample annotations for '{cat_name}'")
                
                # Display sample images with annotations
                for img_id in sample_img_ids:
                    img_file = image_map.get(img_id)
                    anns = annotations_by_image.get(img_id, [])
                    
                    # Filter annotations for this category
                    cat_anns = [ann for ann in anns if ann['category_id'] == cat_id]
                    
                    if not cat_anns:
                        continue
                    
                    # Try to load the image
                    img_path = None
                    if image_dir and img_file:
                        img_path = os.path.join(image_dir, img_file)
                    
                    if img_path and os.path.exists(img_path):
                        try:
                            img = PILImage.open(img_path)
                            img_array = np.array(img)
                            
                            # Draw annotations and display
                            viz_buf = draw_annotations(img_array, cat_anns)
                            st.image(viz_buf, caption=f"Image ID: {img_id}, File: {img_file}")
                            
                            # Show annotation details
                            with st.expander("View annotation details"):
                                st.json(cat_anns)
                        except Exception as e:
                            st.error(f"Error visualizing image {img_file}: {e}")
                    else:
                        st.warning(f"Image file not found: {img_file}")
    else:
        st.warning("No categories found in the dataset.")

def enhanced_category_statistics(coco_data, image_dir=None):
    """
    Display enhanced statistics about categories and their annotations in a COCO format dataset.
    
    Args:
        coco_data (dict): The COCO format data containing categories, annotations, and images.
        image_dir (str, optional): Directory containing the images for visualization.
    """
    if not coco_data:
        st.warning("No COCO data provided for statistics.")
        return
    
    categories = coco_data.get('categories', [])
    annotations = coco_data.get('annotations', [])
    images = coco_data.get('images', [])
    
    # Basic statistics with improved styling
    st.write("## Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Categories", len(categories))
    with col2:
        st.metric("Total Images", len(images))
    with col3:
        st.metric("Total Annotations", len(annotations))
    
    # Count annotations per category
    category_counts = Counter([ann['category_id'] for ann in annotations])
    
    # Create a mapping from category_id to category name
    category_map = {cat['id']: cat['name'] for cat in categories}
    
    # Create a DataFrame for better visualization
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
    
    # Display in two columns with improved styling
    st.write("## Category Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        # Display the statistics table with improved styling
        st.write("### Annotations per Category")
        st.dataframe(stats_df, use_container_width=True)
        
        # Create a bar chart of annotation counts with improved styling
        if not stats_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort by count for better visualization
            sorted_df = stats_df.sort_values('Annotation Count', ascending=False)
            
            # Use a colormap for better visualization
            colors = plt.cm.viridis(np.linspace(0, 0.9, len(sorted_df)))
            bars = ax.bar(sorted_df['Category Name'], sorted_df['Annotation Count'], color=colors)
            
            # Add the values on top of the bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.xlabel('Category', fontweight='bold')
            plt.ylabel('Number of Annotations', fontweight='bold')
            plt.title('Annotations per Category', fontsize=14, fontweight='bold')
            
            # Add grid for better readability
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
    
    with col2:
        # Images per category with improved styling
        st.write("### Images per Category")
        
        if not img_stats_df.empty:
            st.dataframe(img_stats_df, use_container_width=True)
            
            # Create a bar chart of image counts with improved styling
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Sort by count for better visualization
            sorted_df = img_stats_df.sort_values('Image Count', ascending=False)
            
            # Use a different colormap for better distinction
            colors = plt.cm.plasma(np.linspace(0, 0.9, len(sorted_df)))
            bars = ax.bar(sorted_df['Category Name'], sorted_df['Image Count'], color=colors)
            
            # Add the values on top of the bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.xlabel('Category', fontweight='bold')
            plt.ylabel('Number of Images', fontweight='bold')
            plt.title('Images per Category', fontsize=14, fontweight='bold')
            
            # Add grid for better readability
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
    
    # Add visualization of annotation distribution by type
    st.write("## Annotation Type Distribution")
    
    # Count annotation types
    annotation_types = {
        'bbox': sum(1 for ann in annotations if 'bbox' in ann),
        'segmentation': sum(1 for ann in annotations if 'segmentation' in ann and ann['segmentation']),
        'keypoints': sum(1 for ann in annotations if 'keypoints' in ann and ann['keypoints']),
        'caption': sum(1 for ann in annotations if 'caption' in ann),
    }
    
    # Remove types with zero count
    annotation_types = {k: v for k, v in annotation_types.items() if v > 0}
    
    if annotation_types:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create a pie chart for annotation types with improved styling
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # Use a custom colormap for better visualization
            colors = plt.cm.tab10(np.linspace(0, 1, len(annotation_types)))
            
            wedges, texts, autotexts = ax.pie(
                annotation_types.values(), 
                labels=annotation_types.keys(),
                autopct='%1.1f%%',
                startangle=90,
                shadow=True,
                explode=[0.05] * len(annotation_types),  # Slight explode for all slices
                colors=colors,
                textprops={'fontsize': 12, 'weight': 'bold'}
            )
            plt.setp(autotexts, size=10, weight='bold')
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title('Distribution of Annotation Types', fontsize=16, fontweight='bold')
            
            st.pyplot(fig)
        
        with col2:
            # Display annotation type counts as a table
            type_df = pd.DataFrame({
                'Annotation Type': list(annotation_types.keys()),
                'Count': list(annotation_types.values())
            })
            st.dataframe(type_df, use_container_width=True)
    else:
        st.info("No annotation type information available.")
    
    # Add visualization of sample annotations
    st.write("## Sample Visualizations")
    if st.button("Show Sample Annotations"):
        visualize_coco_annotations_sample(coco_data, image_dir=image_dir)
    
    # Add annotation size distribution
    st.write("## Annotation Size Distribution")
    
    # Extract bounding box sizes
    if any('bbox' in ann for ann in annotations):
        bbox_areas = []
        for ann in annotations:
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                area = w * h
                bbox_areas.append({
                    'category_id': ann['category_id'],
                    'category_name': category_map.get(ann['category_id'], 'Unknown'),
                    'area': area
                })
        
        if bbox_areas:
            bbox_df = pd.DataFrame(bbox_areas)
            
            # Create histogram of bbox areas
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(bbox_df['area'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel('Bounding Box Area (pixels²)', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title('Distribution of Bounding Box Sizes', fontsize=14, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
            
            # Show average bbox area by category
            st.write("### Average Bounding Box Area by Category")
            avg_area_by_cat = bbox_df.groupby('category_name')['area'].mean().reset_index()
            avg_area_by_cat = avg_area_by_cat.sort_values('area', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(avg_area_by_cat['category_name'], avg_area_by_cat['area'], 
                         color=plt.cm.cool(np.linspace(0, 0.9, len(avg_area_by_cat))))
            
            # Add the values on top of the bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.xlabel('Category', fontweight='bold')
            plt.ylabel('Average Area (pixels²)', fontweight='bold')
            plt.title('Average Bounding Box Area by Category', fontsize=14, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            st.pyplot(fig)
