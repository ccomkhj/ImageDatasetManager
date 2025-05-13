import streamlit as st
import datumaro as dm
from datumaro.components.dataset import Dataset
import os
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import tempfile
from utils import show_category_statistics


def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.
    Boxes are in format [x1, y1, x2, y2].
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 < x1 or y2 < y1:
        return 0.0  # No intersection

    intersection_area = (x2 - x1) * (y2 - y1)

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0

    return iou


def match_images_by_filename(coco1, coco2):
    """
    Match images from two COCO datasets by filename.
    Returns a dictionary mapping image_id from coco1 to image_id from coco2.
    """
    # Create dictionaries mapping filename to image_id for both datasets
    coco1_filename_to_id = {img["file_name"]: img["id"] for img in coco1["images"]}
    coco2_filename_to_id = {img["file_name"]: img["id"] for img in coco2["images"]}

    # Create a mapping from coco1 image_id to coco2 image_id
    image_id_mapping = {}
    for img in coco1["images"]:
        filename = img["file_name"]
        if filename in coco2_filename_to_id:
            image_id_mapping[img["id"]] = coco2_filename_to_id[filename]

    return image_id_mapping


def create_category_mapping(coco1, coco2):
    """
    Create a mapping between category IDs in two COCO datasets based on category names.
    Returns a dictionary mapping category_id from coco1 to category_id from coco2.
    """
    # Create dictionaries mapping category name to category_id for both datasets
    coco1_name_to_id = {cat["name"]: cat["id"] for cat in coco1["categories"]}
    coco2_name_to_id = {cat["name"]: cat["id"] for cat in coco2["categories"]}

    # Create a mapping from coco1 category_id to coco2 category_id
    category_id_mapping = {}
    for cat in coco1["categories"]:
        name = cat["name"]
        if name in coco2_name_to_id:
            category_id_mapping[cat["id"]] = coco2_name_to_id[name]

    return category_id_mapping


def compare_coco_annotations(coco1, coco2, iou_threshold=0.7):
    """
    Compare two COCO annotation files and find bounding boxes with high IoU but different categories.

    Args:
        coco1: First COCO dataset
        coco2: Second COCO dataset
        iou_threshold: IoU threshold for matching bounding boxes

    Returns:
        List of dictionaries containing information about mismatched annotations
    """
    # Check if both datasets have the same categories
    coco1_categories = {cat["id"]: cat["name"] for cat in coco1["categories"]}
    coco2_categories = {cat["id"]: cat["name"] for cat in coco2["categories"]}

    # Match images by filename
    image_id_mapping = match_images_by_filename(coco1, coco2)

    # Create a mapping between category IDs
    category_id_mapping = create_category_mapping(coco1, coco2)

    # Create a dictionary mapping image_id to annotations for coco2
    coco2_image_to_anns = {}
    for ann in coco2["annotations"]:
        image_id = ann["image_id"]
        if image_id not in coco2_image_to_anns:
            coco2_image_to_anns[image_id] = []
        coco2_image_to_anns[image_id].append(ann)

    # Find mismatched annotations
    mismatches = []

    for ann1 in coco1["annotations"]:
        coco1_image_id = ann1["image_id"]

        # Skip if this image doesn't exist in coco2
        if coco1_image_id not in image_id_mapping:
            continue

        coco2_image_id = image_id_mapping[coco1_image_id]

        # Skip if this image has no annotations in coco2
        if coco2_image_id not in coco2_image_to_anns:
            continue

        # Get the category name for this annotation in coco1
        coco1_category_id = ann1["category_id"]
        if coco1_category_id not in coco1_categories:
            continue
        coco1_category_name = coco1_categories[coco1_category_id]

        # Get the corresponding category ID in coco2
        if coco1_category_id not in category_id_mapping:
            continue
        expected_coco2_category_id = category_id_mapping[coco1_category_id]

        # Convert bbox format from [x, y, width, height] to [x1, y1, x2, y2]
        bbox1 = ann1["bbox"]
        box1 = [bbox1[0], bbox1[1], bbox1[0] + bbox1[2], bbox1[1] + bbox1[3]]

        # Compare with all annotations for this image in coco2
        for ann2 in coco2_image_to_anns[coco2_image_id]:
            bbox2 = ann2["bbox"]
            box2 = [bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3]]

            # Calculate IoU
            iou = calculate_iou(box1, box2)

            # If IoU is above threshold but categories are different
            if (
                iou >= iou_threshold
                and ann2["category_id"] != expected_coco2_category_id
            ):
                # Find the image filename
                image_filename = None
                for img in coco1["images"]:
                    if img["id"] == coco1_image_id:
                        image_filename = img["file_name"]
                        break

                # Get the category name for the annotation in coco2
                coco2_category_id = ann2["category_id"]
                coco2_category_name = coco2_categories.get(coco2_category_id, "Unknown")

                mismatches.append(
                    {
                        "image_id": coco1_image_id,
                        "image_filename": image_filename,
                        "box1": box1,
                        "box2": box2,
                        "iou": iou,
                        "coco1_category": coco1_category_name,
                        "coco2_category": coco2_category_name,
                        "ann1": ann1,
                        "ann2": ann2,
                    }
                )

    return mismatches


def draw_dashed_rectangle(draw, box, color, width=5, dash_length=10, space_length=5):
    """
    Draw a dashed rectangle on the image.

    Args:
        draw: ImageDraw object
        box: Rectangle coordinates [x1, y1, x2, y2]
        color: Line color
        width: Line width
        dash_length: Length of each dash
        space_length: Length of space between dashes
    """
    x1, y1, x2, y2 = box

    # Draw top line
    x, y = x1, y1
    while x < x2:
        segment_length = min(dash_length, x2 - x)
        draw.line([(x, y), (x + segment_length, y)], fill=color, width=width)
        x += dash_length + space_length

    # Draw right line
    x, y = x2, y1
    while y < y2:
        segment_length = min(dash_length, y2 - y)
        draw.line([(x, y), (x, y + segment_length)], fill=color, width=width)
        y += dash_length + space_length

    # Draw bottom line
    x, y = x2, y2
    while x > x1:
        segment_length = min(dash_length, x - x1)
        draw.line([(x, y), (x - segment_length, y)], fill=color, width=width)
        x -= dash_length + space_length

    # Draw left line
    x, y = x1, y2
    while y > y1:
        segment_length = min(dash_length, y - y1)
        draw.line([(x, y), (x, y - segment_length)], fill=color, width=width)
        y -= dash_length + space_length


def visualize_mismatches(images_dir, mismatches):
    """
    Visualize images with mismatched annotations.

    Args:
        images_dir: Directory containing the images
        mismatches: List of mismatched annotations
    """
    if not mismatches:
        st.warning("No mismatches found.")
        return

    # Define colors for visualization
    COCO1_COLOR = (255, 0, 0)  # Red for COCO1
    COCO2_COLOR = (0, 0, 255)  # Blue for COCO2

    # Show color legend
    st.write("### Visualization Legend")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"<div style='background-color:rgb{COCO1_COLOR};color:white;padding:5px;border-radius:5px;text-align:center;font-weight:bold;'>COCO1 - Solid Line</div>",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"<div style='background-color:rgb{COCO2_COLOR};color:white;padding:5px;border-radius:5px;text-align:center;font-weight:bold;'>COCO2 - Dashed Line</div>",
            unsafe_allow_html=True,
        )

    # Group mismatches by image
    mismatches_by_image = {}
    for mismatch in mismatches:
        image_id = mismatch["image_id"]
        if image_id not in mismatches_by_image:
            mismatches_by_image[image_id] = []
        mismatches_by_image[image_id].append(mismatch)

    # Visualize each image with its mismatches
    for image_id, image_mismatches in mismatches_by_image.items():
        if not image_mismatches:
            continue

        # Get the image filename
        image_filename = image_mismatches[0]["image_filename"]
        image_path = os.path.join(images_dir, image_filename)

        if not os.path.exists(image_path):
            st.warning(f"Image not found: {image_path}")
            continue

        # Load the image
        try:
            img = Image.open(image_path)
            draw = ImageDraw.Draw(img)

            # Prepare data for the dataframe
            mismatch_data = []

            # Draw each mismatched box
            for mismatch in image_mismatches:
                # Draw box from coco1 in red with solid line (thicker)
                box1 = mismatch["box1"]
                draw.rectangle(box1, outline=COCO1_COLOR, width=5)

                # Draw box from coco2 in blue with dashed line
                box2 = mismatch["box2"]
                draw_dashed_rectangle(draw, box2, COCO2_COLOR, width=5)

                # Draw labels
                label1 = f"{mismatch['coco1_category']}"
                label2 = f"{mismatch['coco2_category']}"
                # Draw labels with larger font and better visibility
                try:
                    # Try to load a larger font (adjust size as needed)
                    try:
                        font = ImageFont.truetype("arial.ttf", 48)  # Try Arial first
                    except:
                        try:
                            font = ImageFont.truetype(
                                "LiberationSans-Regular.ttf", 48
                            )  # Try Linux alternative
                        except:
                            font = ImageFont.load_default(
                                size=48
                            )  # Fallback to default with larger size

                    # Calculate text positions
                    text1_pos = (
                        box1[0],
                        box1[1] - 40,
                    )  # Higher position for COCO1 label
                    text2_pos = (
                        box2[0],
                        box2[3] + 10,
                    )  # Higher position for COCO2 label

                    # Draw text with background for better visibility
                    # COCO1 label
                    text_bbox1 = draw.textbbox(text1_pos, label1, font=font)
                    padding = 5
                    draw.rectangle(
                        [
                            text_bbox1[0] - padding,
                            text_bbox1[1] - padding,
                            text_bbox1[2] + padding,
                            text_bbox1[3] + padding,
                        ],
                        fill=(50, 50, 50),  # Dark gray background
                    )
                    draw.text(text1_pos, label1, fill=COCO1_COLOR, font=font)

                    # COCO2 label
                    text_bbox2 = draw.textbbox(text2_pos, label2, font=font)
                    draw.rectangle(
                        [
                            text_bbox2[0] - padding,
                            text_bbox2[1] - padding,
                            text_bbox2[2] + padding,
                            text_bbox2[3] + padding,
                        ],
                        fill=(50, 70, 50),
                    )
                    draw.text(text2_pos, label2, fill=COCO2_COLOR, font=font)

                except Exception as e:
                    st.warning(f"Couldn't use preferred font: {e}")
                    # Fallback if font loading fails
                    draw.text((box1[0], box1[1] - 20), label1, fill=COCO1_COLOR)
                    draw.text((box2[0], box2[1] - 40), label2, fill=COCO2_COLOR)

                # Add data for the dataframe
                mismatch_data.append(
                    {
                        "COCO1 Category": mismatch["coco1_category"],
                        "COCO2 Category": mismatch["coco2_category"],
                        "IoU": f"{mismatch['iou']:.2f}",
                    }
                )

            # Display the image with caption
            st.image(img, caption=f"Image: {image_filename}")

            # Display mismatch details as a dataframe
            if mismatch_data:
                st.write("#### Mismatch Details:")
                df = pd.DataFrame(mismatch_data)
                st.dataframe(df)

        except Exception as e:
            st.error(f"Error visualizing image {image_filename}: {e}")


def save_uploaded_coco_file(uploaded_file):
    """Save an uploaded COCO file to a temporary directory and return the path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create annotations directory
        annotations_dir = os.path.join(temp_dir, "annotations")
        os.makedirs(annotations_dir, exist_ok=True)

        # Save the uploaded file
        file_path = os.path.join(annotations_dir, "instances_default.json")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return temp_dir, file_path


def save_uploaded_images(uploaded_images, temp_dir):
    """Save uploaded images to a temporary directory and return the images directory path."""
    # Create images directory
    images_dir = os.path.join(temp_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Save each uploaded image
    for image in uploaded_images:
        image_path = os.path.join(images_dir, image.name)
        with open(image_path, "wb") as f:
            f.write(image.getbuffer())

    return images_dir


def main():
    st.write("# Compare COCO Annotations")

    # Upload images
    st.write("## Upload Images")
    images = st.file_uploader(
        "Upload Image Files (used for both COCO files)",
        accept_multiple_files=True,
        type=["jpg", "jpeg", "png"],
    )

    # Upload COCO files
    st.write("## Upload COCO Files")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### COCO File 1")
        coco1_file = st.file_uploader(
            "Upload First COCO File", type=["json"], key="coco1"
        )

    with col2:
        st.write("### COCO File 2")
        coco2_file = st.file_uploader(
            "Upload Second COCO File", type=["json"], key="coco2"
        )

    # IoU threshold slider
    iou_threshold = st.slider(
        "IoU Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Bounding boxes with IoU above this threshold will be compared",
    )

    if st.button("Compare Annotations"):
        if not images:
            st.warning("Please upload images.")
            return

        if not coco1_file or not coco2_file:
            st.warning("Please upload both COCO files.")
            return

        # Save uploaded files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save images
            images_dir = save_uploaded_images(images, temp_dir)

            # Load COCO files
            try:
                coco1_data = json.load(coco1_file)
                coco2_data = json.load(coco2_file)
            except Exception as e:
                st.error(f"Error loading COCO files: {e}")
                return

            # Check if both datasets have categories
            if "categories" not in coco1_data or "categories" not in coco2_data:
                st.error("Both COCO files must have categories.")
                return

            # Check if both datasets have images
            if "images" not in coco1_data or "images" not in coco2_data:
                st.error("Both COCO files must have images.")
                return

            # Check if both datasets have annotations
            if "annotations" not in coco1_data or "annotations" not in coco2_data:
                st.error("Both COCO files must have annotations.")
                return

            # Show category statistics for both datasets
            st.write("## Category Statistics")

            col1, col2 = st.columns(2)

            with col1:
                st.write("### COCO File 1")
                show_category_statistics(coco1_data)

            with col2:
                st.write("### COCO File 2")
                show_category_statistics(coco2_data)

            # Compare annotations
            st.write("## Comparison Results")
            mismatches = compare_coco_annotations(coco1_data, coco2_data, iou_threshold)

            if not mismatches:
                st.success(
                    "No mismatches found! All overlapping bounding boxes have matching categories."
                )
            else:
                st.warning(
                    f"Found {len(mismatches)} mismatches where bounding boxes overlap (IoU ≥ {iou_threshold}) but have different categories."
                )

                # Visualize mismatches
                st.write("## Visualization of Mismatches")
                visualize_mismatches(images_dir, mismatches)


if __name__ == "__main__":
    main()
