# Image Dataset Manager

A streamlined GUI for computer vision dataset management using the Datumaro framework. Simplifies dataset registration, merging, filtering, and validation with an intuitive interface designed for researchers and engineers.

## Features

- **Register** new datasets with automatic train/validation splitting
- **Merge** multiple datasets while maintaining category consistency
- **Filter** annotations using custom Python functions
- **Validate** datasets with comprehensive error detection
- **Manage** annotation categories with an intuitive editor
- **Visualize** annotations with advanced statistics and insights
- **Compare** annotations between different versions of a dataset
- **S3 Integration** for cloud-based dataset management

## Project Structure

```
ImageDatasetManager/
├── app.py                 # Main application entry point
├── utils.py               # Utility functions
├── enhanced_viz.py        # Enhanced visualization module
├── pages/
│   ├── new.py             # Register new datasets
│   ├── merge.py           # Merge existing datasets
│   ├── filter.py          # Filter annotations
│   ├── validate.py        # Validate datasets
│   ├── category.py        # Manage categories
│   ├── stats_visualizer.py # Advanced statistics visualization
│   └── compare.py         # Compare annotation versions
├── requirements.txt
└── credentials/
    └── aws.yaml           # AWS credentials for S3 access
```

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ccomkhj/ImageDatasetManager
    cd ImageDatasetManager/
    ```

2. Create a virtual environment and activate it:
    ```bash
    conda create -n ImageDatasetManager python=3.11 -y
    conda activate ImageDatasetManager
    ```
    [Note] tested with python =< 3.11
    OR
    ```bash
    python3 -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. (if you want to use S3) Set up AWS credentials in `credentials/aws.yaml`:
    ```yaml
    aws_access_key_id: <YOUR_AWS_ACCESS_KEY_ID>
    aws_secret_access_key: <YOUR_AWS_SECRET_ACCESS_KEY>
    ```

## Usage

1. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. Open your browser and go to `http://localhost:8501` to interact with the Datumaro-GUI.


## To read new dataset
```
-- annotations
    |- instances_train.json
    |- instances_val.json
    # subsets are train and val
-- images
    |- train
    |- val

```
or
```
-- annotations
    |- instances_default.json
    # subsets are only default
-- images
    |- default
```

## Key Interfaces

### Register Annotation

Upload and process images and annotations with automatic train/val splitting, visualization, and S3 integration.

### Merge Annotations

Combine datasets from S3 or local sources with category consistency checking and customizable splitting options.

### Filter Annotation

Apply custom Python filters with a code editor, templates, and visualization of filtered results.

### Validate Dataset

Identify issues with comprehensive validation reports, statistics, and visual summaries.

### Category Editor

Manage annotation categories with an interactive table editor and visualization tools.

### Statistics Visualizer

Advanced visualization interface with:

- Interactive category-based annotation samples
- Comprehensive distribution charts and statistics
- Annotation type and size analysis
- Support for local uploads, paths, and S3 sources
- Detailed visualization of bounding boxes, segmentation, and keypoints

### Compare Annotations

Compare two COCO annotation files to identify discrepancies:

- Upload two COCO annotation files and their corresponding images
- Identify bounding boxes with high IoU but different categories
- Visualize mismatches with color-coded bounding boxes (COCO1: red solid lines, COCO2: blue dashed lines)
- Adjust IoU threshold for comparison sensitivity
- View detailed mismatch information in a structured dataframe
- Ideal for quality control and annotation verification


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.