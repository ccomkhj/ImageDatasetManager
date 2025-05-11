# Datumaro-GUI

Datumaro-GUI is a powerful graphical user interface (GUI) built with Streamlit that streamlines working with computer vision datasets using the Datumaro framework. Designed for computer vision researchers and engineers, this tool simplifies dataset management tasks with an intuitive interface for registering, merging, filtering, and validating annotations.

## Features

- **Register new datasets**: Upload and process images and annotations to create new datasets with automatic train/validation splitting.
- **Merge datasets**: Combine existing datasets with new data and intelligently split them into training and validation sets.
- **Filter annotations**: Apply custom Python filters to datasets to extract specific data based on sophisticated conditions.
- **Validate annotations**: Identify and visualize potential issues in your datasets with comprehensive validation reports.
- **Category management**: Edit, merge, and manage annotation categories with an intuitive interface.
- **AWS S3 Integration**: Seamlessly load existing datasets from S3 and upload the processed datasets back to S3.
- **Visualization tools**: Visualize annotations and view detailed category statistics for better dataset understanding.

## Project Structure

```
datumaro-gui/
├── app.py
├── utils.py
├── pages/
│   ├── merge.py
│   ├── new.py
│   ├── filter.py
│   ├── validate.py
│   └── category.py
├── requirements.txt
├── README.md
└── credentials/
    └── aws.yaml
```

- **app.py**: The main entry point for the Streamlit application.
- **utils.py**: Utility functions for handling file uploads, AWS S3 interactions, visualization, and other helper functions.
- **pages/**: Contains the different pages for the Streamlit app.
    - **new.py**: Page for registering new annotations with automatic train/val splitting.
    - **merge.py**: Page for merging multiple datasets with customizable options.
    - **filter.py**: Page for applying custom Python filters to datasets.
    - **validate.py**: Page for validating datasets and identifying potential issues.
    - **category.py**: Page for managing and editing annotation categories.
- **credentials/aws.yaml**: Stores the AWS credentials for interacting with S3.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ccomkhj/datumaro-gui.git
    cd datumaro-gui/
    ```

2. Create a virtual environment and activate it:
    ```bash
    conda create -n datumaro-gui python=3.11 -y
    conda activate datumaro-gui
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
    |- instance_default.json
    # subsets are only default
-- images
    |- train
```

## Recommended Workflow

1. Use **Register Annotation** (`new`) if it's your first dataset
2. Use **Merge Annotations** (`merge`) if you need to combine datasets
3. Use **Filter Annotation** (`filter`) to create datasets selectively (after `merge` or `new`)
4. Use **Validate Dataset** (`validate`) to check for potential issues in your dataset
5. Use **Category Editor** (`category`) to manage and edit annotation categories

## Interface Overview

### Register Annotation

The Register Annotation interface allows you to upload image files and annotation files to create a new dataset. The interface includes:

- File upload areas for images and annotation files
- Dropdown menus for selecting annotation type (instances, keypoints, segmentation)
- Dataset type selection (COCO, VOC, etc.)
- Automatic train/validation splitting (80/20 by default)
- Visualization tools to preview annotations
- Category statistics display with distribution charts
- S3 upload functionality with custom URI and comments

### Merge Annotations

The Merge Annotations interface combines multiple datasets with these features:

- Source selection between S3 and local datasets
- Multiple dataset merging with category consistency checking
- Customizable train/validation splitting options
- Visualization of merged annotations
- Category statistics before and after merging
- S3 upload functionality for the merged dataset

### Filter Annotation

The Filter Annotation interface provides powerful dataset filtering with:

- Custom Python filter function editor with syntax highlighting
- Sample filter code templates for common filtering operations
- Dataset statistics before and after filtering
- Visualization of filtered annotations
- S3 upload functionality for filtered datasets

### Validate Dataset

The Validate Dataset interface helps identify issues in your datasets with:

- Support for local and S3 dataset sources
- Comprehensive validation reports with severity indicators
- Dataset statistics and metrics
- Visual summary of validation results
- Detailed anomaly descriptions and item IDs

### Category Editor

The Category Editor interface provides tools for managing annotation categories:

- Interactive table for editing category properties
- Category statistics visualization
- Tools to remove categories and their associated annotations
- Export functionality for modified annotation files


## Use Cases for Computer Vision Experts

### Dataset Preparation for Model Training

Datumaro-GUI streamlines the process of preparing datasets for training computer vision models:

- **Data Consolidation**: Easily merge multiple data sources while maintaining category consistency
- **Train/Val Splitting**: Automatically create properly balanced training and validation sets
- **Custom Filtering**: Apply sophisticated filters to create specialized datasets for specific model training needs
- **Category Management**: Standardize categories across datasets for consistent model training

### Dataset Quality Assurance

Ensure your datasets meet quality standards before training:

- **Validation Reports**: Identify and fix annotation issues before they affect model training
- **Category Statistics**: Analyze class distributions to identify imbalances
- **Visualization**: Visually inspect annotations to catch labeling errors

### Efficient Dataset Management

Streamline your dataset workflow:

- **S3 Integration**: Seamlessly work with cloud-stored datasets
- **Batch Processing**: Process entire datasets with a few clicks
- **Standardization**: Ensure consistent dataset formats across projects

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue on our GitHub repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.