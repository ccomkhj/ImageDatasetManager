import streamlit as st

st.set_page_config(page_title="Datumaro-GUI📊", layout="wide", page_icon="🌟")

# Streamlit App Title and Description
st.title("Datumaro-GUI📊")
st.markdown(
    """
Welcome to **Datumaro-GUI**! 🎉

Datumaro-GUI provides an easy-to-use graphical interface for managing computer vision datasets using Datumaro. 🚀
"""
)

# Sidebar Information
st.sidebar.title("Navigation")
st.sidebar.markdown("🔧 **New Task:** Register a new annotation.")
st.sidebar.markdown(
    "🔄 **Merge Datasets:** Merge new annotations with existing datasets."
)

# Additional Sidebar Information
st.sidebar.title("About")
st.sidebar.info(
    """
Datumaro-GUI is built to simplify the process of managing datasets in computer vision tasks using Datumaro, 
by offering a user-friendly graphical interface. 

Check out the [GitHub repository](https://github.com/ccomkhj/datumaro-gui) for more information.
"""
)

# Displaying links with icons
st.markdown(
    """
## Overview and Features ✨

- **Register New Annotations** 📝: Easily upload and process images and annotations to create new datasets.
- **Merge Datasets** 🔀: Combine existing datasets with new data, and split them into training and validation sets.
- **AWS S3 Integration** ☁️: Load existing datasets from S3 and upload processed datasets back to S3.

Use the options in the sidebar to get started!
"""
)
